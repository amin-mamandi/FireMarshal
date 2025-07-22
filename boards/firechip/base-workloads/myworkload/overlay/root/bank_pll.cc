/**
 * mlp: memory-level-parallelism (MLP) detector
 *
 * Copyright (C) 2015  Heechul Yun <heechul.yun@ku.edu> 
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 */ 

/**************************************************************************
 * Conditional Compilation Options
 **************************************************************************/
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <signal.h>
#include <m5ops.h>

/**************************************************************************
 * Included Files
 **************************************************************************/
#include <iostream>     // std::cout
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

#include <sched.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

// Global variables
static int num_attackers = 0;
static int is_attacker = 0;
static int sync_enabled = 0;
static int sync_id = -1;

static volatile int keep_running = 1;

void signal_handler(int sig) {
    keep_running = 0;
    printf("CPU%d: Received termination signal\n", sched_getcpu());
    fflush(stdout);
}

// Initialize synchronization
void init_sync(int id, int attackers, int is_attack) {
    if (!sync_enabled) return;
    
    is_attacker = is_attack;
    num_attackers = attackers;
    sync_id = id;
    
    printf("Process %d: Initialized file-based synchronization (is_attacker=%d, total=%d)\n", 
           sync_id, is_attacker, num_attackers + 1);
}

// Synchronize before measurement using file-based approach
void sync_before_measurement() {
    if (!sync_enabled) return;
    
    char filename[100];
    sprintf(filename, "/tmp/barrier_%d", sync_id);
    
    // Signal that this process is ready
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "%d", sync_id);
        fclose(f);
        printf("Process %d: Created ready file %s\n", sync_id, filename);
    } else {
        printf("Process %d: Failed to create file %s: %s\n", sync_id, filename, strerror(errno));
    }
    
    printf("Process %d: Waiting for all processes (%d total)...\n", sync_id, num_attackers + 1);
    
    // Wait for all processes to be ready
    int count = 0;
    int max_attempts = 1000; // Avoid infinite loops
    int attempt = 0;
    
    while (count < num_attackers + 1 && attempt < max_attempts) {
        count = 0;
        for (int i = 0; i <= num_attackers; i++) {
            sprintf(filename, "/tmp/barrier_%d", i);
            if (access(filename, F_OK) == 0) {
                count++;
            }
        }
        
        if (count < num_attackers + 1) {
            usleep(10000); // 10ms delay to avoid tight polling
        }
        attempt++;
    }
    
    printf("Process %d: All processes ready (%d/%d)\n", sync_id, count, num_attackers + 1);
    
    // If we're the victim process, take checkpoint and reset stats
    if (!is_attacker) {
        printf("All processes ready, taking checkpoint...\n");

        m5_checkpoint(0, 0);

        printf("Checkpoint taken, waiting 70ms before continuing...\n");
        
        // Give attackers time to warm up caches
        usleep(70000); // 100ms delay
        printf("Victim: Continuing after warmup period\n");
    } else {
        // Attackers continue immediately after all processes are ready
        printf("Process %d: Attacker continuing\n", sync_id);
    }
}

/**************************************************************************
 * Public Definitions
 **************************************************************************/
#define MAX_MLP 256
#define CACHE_LINE_SIZE 64
#ifdef __arm__
#  define DEFAULT_ALLOC_SIZE_KB 4096
#else
#  define DEFAULT_ALLOC_SIZE_KB 16384
#endif
#define DEFAULT_ITER 100

#ifdef __LP64__
#define BITS_PER_LONG 64
#else
#define BITS_PER_LONG 32
#endif
#define BITOP_WORD(nr)		((nr) / BITS_PER_LONG)
#define PAGE_SHIFT 12

#define MAX_COLORS 64

/**************************************************************************
 * Public Types
 **************************************************************************/
enum access_type { READ, WRITE};

/**************************************************************************
 * Global Variables
 **************************************************************************/
static int g_mem_size = (DEFAULT_ALLOC_SIZE_KB*1024);
static int* list[MAX_MLP];
static int next[MAX_MLP];

static int g_debug = 0;
static int g_color[MAX_COLORS]; // not assigned
static int g_color_cnt = 0;

// static unsigned long dram_bitmask = 0x1e000; // 16|15,14,13,--| : DDR4
static unsigned long dram_bitmask = 0x78000;  // --,14,13,12|11  : HBM2
static unsigned long channel_bitmask = 0x380; // Channel bits [7:9] for 8 channels
static unsigned long pseudo_channel_bitmask = 0x40; // Pseudo-channel bit [6] for HBM

static int g_channel[MAX_COLORS]; // Target channels
static int g_channel_cnt = 0;

static int g_pseudo_channel[MAX_COLORS]; // Target pseudo-channels
static int g_pseudo_channel_cnt = 0;
static int attack_type = 0;  // 0=pointer chasing, 1=matrix multiplication
static float *matrix_A = NULL;
static float *matrix_B = NULL;
static float *matrix_C = NULL;
static int matrix_dimension = 512;  // Default matrix size

/**************************************************************************
 * Public Function Prototypes
 **************************************************************************/
uint64_t get_elapsed(struct timespec *start, struct timespec *end)
{
	uint64_t dur;

	dur = ((uint64_t)end->tv_sec * 1000000000 + end->tv_nsec) - 
		((uint64_t)start->tv_sec * 1000000000 + start->tv_nsec);
	return dur;
}

// ----------------------------------------------
long utime()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);

	return (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000;
}

uint64_t nstime()
{
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return ts.tv_sec * 1000000000 + ts.tv_nsec;
}


static __always_inline unsigned long __ffs(unsigned long word)
{
        int num = 0;

#if BITS_PER_LONG == 64
        if ((word & 0xffffffff) == 0) {
                num += 32;
                word >>= 32;
        }
#endif
        if ((word & 0xffff) == 0) {
                num += 16;
                word >>= 16;
        }
        if ((word & 0xff) == 0) {
                num += 8;
                word >>= 8;
        }
        if ((word & 0xf) == 0) {
                num += 4;
                word >>= 4;
        }
        if ((word & 0x3) == 0) {
                num += 2;
                word >>= 2;
        }
        if ((word & 0x1) == 0)
                num += 1;
        return num;
}

/*
 * Find the next set bit in a memory region.
 */
unsigned long find_next_bit(const unsigned long *addr, unsigned long size,
			    unsigned long offset)
{
	const unsigned long *p = addr + BITOP_WORD(offset);
	unsigned long result = offset & ~(BITS_PER_LONG-1);
	unsigned long tmp;

	if (offset >= size)
		return size;
	size -= result;
	offset %= BITS_PER_LONG;
	if (offset) {
		tmp = *(p++);
		tmp &= (~0UL << offset);
		if (size < BITS_PER_LONG)
			goto found_first;
		if (tmp)
			goto found_middle;
		size -= BITS_PER_LONG;
		result += BITS_PER_LONG;
	}
	while (size & ~(BITS_PER_LONG-1)) {
		if ((tmp = *(p++)))
			goto found_middle;
		result += BITS_PER_LONG;
		size -= BITS_PER_LONG;
	}
	if (!size)
		return result;
	tmp = *p;

found_first:
	tmp &= (~0UL >> (BITS_PER_LONG - size));
	if (tmp == 0UL)		/* Are any bits set? */
		return result + size;	/* Nope. */
found_middle:
	return result + __ffs(tmp);
}

#define find_first_bit(addr, size) find_next_bit((addr), (size), 0)

#define for_each_set_bit(bit, addr, size) \
	for ((bit) = find_first_bit((addr), (size));		\
	     (bit) < (size);					\
	     (bit) = find_next_bit((addr), (size), (bit) + 1))

int paddr_to_color(unsigned long mask, unsigned long paddr)
{
	int color = 0;
	int idx = 0;
	unsigned long c = 0;
	for_each_set_bit(c, &mask, BITS_PER_LONG) {
		if ((paddr >> (c)) & 0x1)
			color |= (1<<idx);
		idx++;
	}
	return color;
}

void print_final_bandwidth(int cpuid, uint64_t nsdiff, uint64_t accesses, int access_type, int attack_type, int ws) {
	
	double  avglat = (double)nsdiff/accesses;

	printf("alloc. size: %d (%d KB)\n", g_mem_size, g_mem_size/1024);
	int total_ws =  ws * CACHE_LINE_SIZE;
	printf("ws size: %d (%d KB)\n", total_ws, total_ws / 1024);
	printf("duration %.0f ns, #access %ld\n", (double)nsdiff, accesses);
	printf("Avg. latency %.2f ns\n", avglat);	
	// printf("CPU%d: bandwidth %.2f MB/s\n", cpuid, (double)64*1000*accesses/nsdiff);
	if (access_type == READ) {
		printf("CPU%d: Final READ bandwidth %.2f MB/s\n", cpuid, (double)64*1000*accesses/nsdiff);
	} else {
		printf("CPU%d: Final WRITE bandwidth %.2f MB/s\n", cpuid, (double)64*1000*accesses/nsdiff);
	}

    fflush(stdout);
}

// Matrix multiplication functions
void init_matrix_data(float *A, float *B, float *C, int dimension) {
    srand(292);
    for(int i = 0; i < dimension; i++) {
        for(int j = 0; j < dimension; j++) {
            A[dimension*i+j] = (float)rand()/(float)(RAND_MAX) - 0.5;
            B[dimension*i+j] = (float)rand()/(float)(RAND_MAX) - 0.5;
            C[dimension*i+j] = 0.0;
        }
    }
}

double matrix_checksum(float *C, int dimension) {
    double sum = 0.0;
    for(int i = 0; i < dimension; i++) {
        for(int j = 0; j < dimension; j++) {
            sum += C[i*dimension+j];
        }
    }
    return sum;
}

// Matrix multiplication with jk order switch (opt1 from matrix.c)
int64_t run_matrix_multiplication_opt1() {
    int64_t total_operations = 0;
    
    // Matrix multiplication with jk order switch
    for(int i = 0; i < matrix_dimension; i++) {
        for(int k = 0; k < matrix_dimension; k++) {
            for(int j = 0; j < matrix_dimension; j++) {
                matrix_C[matrix_dimension*i+j] += (matrix_A[matrix_dimension*i+k] * matrix_B[matrix_dimension*k+j]);
            }
        }
    }
    
    // Count operations (3 memory accesses per inner loop iteration)
    total_operations = (int64_t)matrix_dimension * matrix_dimension * matrix_dimension * 3;
    
    return total_operations;
}


int64_t run_matrix_multiplication_opt0()
{
	int64_t total_operations = 0;

    for(int i = 0; i < matrix_dimension; i++) {
        for(int j = 0; j < matrix_dimension; j++) {
            for(int k = 0; k < matrix_dimension; k++) {
				matrix_C[matrix_dimension*i+j] += (matrix_A[matrix_dimension*i+k] * matrix_B[matrix_dimension*k+j]);
            }
        }
    }

	// Count operations (3 memory accesses per inner loop iteration)
    total_operations = (int64_t)matrix_dimension * matrix_dimension * matrix_dimension * 3;
    
    return total_operations;	
}

/**************************************************************************
 * Implementation
 **************************************************************************/
int64_t run(int64_t iter, int mlp)
{
	int64_t cnt = 0;

	for (int64_t i = 0; i < iter && keep_running; i++) {
		switch (mlp) {
#define CASE(N) case N: next[N-1] = list[N-1][next[N-1]];
			CASE(256) CASE(255) CASE(254) CASE(253) CASE(252) CASE(251)
			CASE(250) CASE(249) CASE(248) CASE(247) CASE(246) CASE(245)
			CASE(244) CASE(243) CASE(242) CASE(241) CASE(240) CASE(239)
			CASE(238) CASE(237) CASE(236) CASE(235) CASE(234) CASE(233)
			CASE(232) CASE(231) CASE(230) CASE(229) CASE(228) CASE(227)
			CASE(226) CASE(225) CASE(224) CASE(223) CASE(222) CASE(221)
			CASE(220) CASE(219) CASE(218) CASE(217) CASE(216) CASE(215)
			CASE(214) CASE(213) CASE(212) CASE(211) CASE(210) CASE(209)
			CASE(208) CASE(207) CASE(206) CASE(205) CASE(204) CASE(203)
			CASE(202) CASE(201) CASE(200) CASE(199) CASE(198) CASE(197)
			CASE(196) CASE(195) CASE(194) CASE(193) CASE(192) CASE(191)
			CASE(190) CASE(189) CASE(188) CASE(187) CASE(186) CASE(185)
			CASE(184) CASE(183) CASE(182) CASE(181) CASE(180) CASE(179)
			CASE(178) CASE(177) CASE(176) CASE(175) CASE(174) CASE(173)
			CASE(172) CASE(171) CASE(170) CASE(169) CASE(168) CASE(167)
			CASE(166) CASE(165) CASE(164) CASE(163) CASE(162) CASE(161)
			CASE(160) CASE(159) CASE(158) CASE(157) CASE(156) CASE(155)
			CASE(154) CASE(153) CASE(152) CASE(151) CASE(150) CASE(149)
			CASE(148) CASE(147) CASE(146) CASE(145) CASE(144) CASE(143)
			CASE(142) CASE(141) CASE(140) CASE(139) CASE(138) CASE(137)
			CASE(136) CASE(135) CASE(134) CASE(133) CASE(132) CASE(131)
			CASE(130) CASE(129) CASE(128) CASE(127) CASE(126) CASE(125)
			CASE(124) CASE(123) CASE(122) CASE(121) CASE(120) CASE(119)
			CASE(118) CASE(117) CASE(116) CASE(115) CASE(114) CASE(113)
			CASE(112) CASE(111) CASE(110) CASE(109) CASE(108) CASE(107)
			CASE(106) CASE(105) CASE(104) CASE(103) CASE(102) CASE(101)
			CASE(100) CASE(99)  CASE(98)  CASE(97)  CASE(96)  CASE(95)
			CASE(94)  CASE(93)  CASE(92)  CASE(91)  CASE(90)  CASE(89)
			CASE(88)  CASE(87)  CASE(86)  CASE(85)  CASE(84)  CASE(83)
			CASE(82)  CASE(81)  CASE(80)  CASE(79)  CASE(78)  CASE(77)
			CASE(76)  CASE(75)  CASE(74)  CASE(73)  CASE(72)  CASE(71)
			CASE(70)  CASE(69)  CASE(68)  CASE(67)  CASE(66)  CASE(65)
			CASE(64)  CASE(63)  CASE(62)  CASE(61)  CASE(60)  CASE(59)
			CASE(58)  CASE(57)  CASE(56)  CASE(55)  CASE(54)  CASE(53)
			CASE(52)  CASE(51)  CASE(50)  CASE(49)  CASE(48)  CASE(47)
			CASE(46)  CASE(45)  CASE(44)  CASE(43)  CASE(42)  CASE(41)
			CASE(40)  CASE(39)  CASE(38)  CASE(37)  CASE(36)  CASE(35)
			CASE(34)  CASE(33)  CASE(32)  CASE(31)  CASE(30)  CASE(29)
			CASE(28)  CASE(27)  CASE(26)  CASE(25)  CASE(24)  CASE(23)
			CASE(22)  CASE(21)  CASE(20)  CASE(19)  CASE(18)  CASE(17)
			CASE(16)  CASE(15)  CASE(14)  CASE(13)  CASE(12)  CASE(11)
			CASE(10)  CASE(9)   CASE(8)   CASE(7)   CASE(6)   CASE(5)
			CASE(4)   CASE(3)   CASE(2)   CASE(1)
#undef CASE
		}
		cnt += mlp;
	}
	return cnt;
}

long run_write(long iter, int mlp)
{
	long cnt = 0;

	for (long i = 0; i < iter && keep_running; i++) {
		switch (mlp) {
#define CASE(N) case N: list[N-1][next[N-1]+1] = 0xff; next[N-1] = list[N-1][next[N-1]];
			CASE(256) CASE(255) CASE(254) CASE(253) CASE(252) CASE(251)
			CASE(250) CASE(249) CASE(248) CASE(247) CASE(246) CASE(245)
			CASE(244) CASE(243) CASE(242) CASE(241) CASE(240) CASE(239)
			CASE(238) CASE(237) CASE(236) CASE(235) CASE(234) CASE(233)
			CASE(232) CASE(231) CASE(230) CASE(229) CASE(228) CASE(227)
			CASE(226) CASE(225) CASE(224) CASE(223) CASE(222) CASE(221)
			CASE(220) CASE(219) CASE(218) CASE(217) CASE(216) CASE(215)
			CASE(214) CASE(213) CASE(212) CASE(211) CASE(210) CASE(209)
			CASE(208) CASE(207) CASE(206) CASE(205) CASE(204) CASE(203)
			CASE(202) CASE(201) CASE(200) CASE(199) CASE(198) CASE(197)
			CASE(196) CASE(195) CASE(194) CASE(193) CASE(192) CASE(191)
			CASE(190) CASE(189) CASE(188) CASE(187) CASE(186) CASE(185)
			CASE(184) CASE(183) CASE(182) CASE(181) CASE(180) CASE(179)
			CASE(178) CASE(177) CASE(176) CASE(175) CASE(174) CASE(173)
			CASE(172) CASE(171) CASE(170) CASE(169) CASE(168) CASE(167)
			CASE(166) CASE(165) CASE(164) CASE(163) CASE(162) CASE(161)
			CASE(160) CASE(159) CASE(158) CASE(157) CASE(156) CASE(155)
			CASE(154) CASE(153) CASE(152) CASE(151) CASE(150) CASE(149)
			CASE(148) CASE(147) CASE(146) CASE(145) CASE(144) CASE(143)
			CASE(142) CASE(141) CASE(140) CASE(139) CASE(138) CASE(137)
			CASE(136) CASE(135) CASE(134) CASE(133) CASE(132) CASE(131)
			CASE(130) CASE(129) CASE(128) CASE(127) CASE(126) CASE(125)
			CASE(124) CASE(123) CASE(122) CASE(121) CASE(120) CASE(119)
			CASE(118) CASE(117) CASE(116) CASE(115) CASE(114) CASE(113)
			CASE(112) CASE(111) CASE(110) CASE(109) CASE(108) CASE(107)
			CASE(106) CASE(105) CASE(104) CASE(103) CASE(102) CASE(101)
			CASE(100) CASE(99)  CASE(98)  CASE(97)  CASE(96)  CASE(95)
			CASE(94)  CASE(93)  CASE(92)  CASE(91)  CASE(90)  CASE(89)
			CASE(88)  CASE(87)  CASE(86)  CASE(85)  CASE(84)  CASE(83)
			CASE(82)  CASE(81)  CASE(80)  CASE(79)  CASE(78)  CASE(77)
			CASE(76)  CASE(75)  CASE(74)  CASE(73)  CASE(72)  CASE(71)
			CASE(70)  CASE(69)  CASE(68)  CASE(67)  CASE(66)  CASE(65)
			CASE(64)  CASE(63)  CASE(62)  CASE(61)  CASE(60)  CASE(59)
			CASE(58)  CASE(57)  CASE(56)  CASE(55)  CASE(54)  CASE(53)
			CASE(52)  CASE(51)  CASE(50)  CASE(49)  CASE(48)  CASE(47)
			CASE(46)  CASE(45)  CASE(44)  CASE(43)  CASE(42)  CASE(41)
			CASE(40)  CASE(39)  CASE(38)  CASE(37)  CASE(36)  CASE(35)
			CASE(34)  CASE(33)  CASE(32)  CASE(31)  CASE(30)  CASE(29)
			CASE(28)  CASE(27)  CASE(26)  CASE(25)  CASE(24)  CASE(23)
			CASE(22)  CASE(21)  CASE(20)  CASE(19)  CASE(18)  CASE(17)
			CASE(16)  CASE(15)  CASE(14)  CASE(13)  CASE(12)  CASE(11)
			CASE(10)  CASE(9)   CASE(8)   CASE(7)   CASE(6)   CASE(5)
			CASE(4)   CASE(3)   CASE(2)   CASE(1)
#undef CASE
		}
		cnt += mlp;
	}
	return cnt;
}

int main(int argc, char* argv[])
{
	// struct sched_param param;
	cpu_set_t cmask;
	int num_processors;
	int cpuid = 0;

	int *memchunk = NULL;
	int opt, prio;
	int i;

	long repeat = DEFAULT_ITER;
	int mlp = 1;
	int use_hugepage = 0;
	struct timespec start, end;
	int acc_type = READ;

	std::srand (0);
	std::vector<int> myvector;

	/*
	 * get command line options 
	 */
	while ((opt = getopt(argc, argv, "m:a:c:d:e:C:b:B:P:i:l:hxs:n:AS:T:D:")) != -1) {
		switch (opt) {
		case 'm': /* set memory size */
			g_mem_size = 1024 * strtol(optarg, NULL, 0);
			break;
		case 'a': /* set access type */
			if (!strcmp(optarg, "read"))
				acc_type = READ;
			else if (!strcmp(optarg, "write"))
				acc_type = WRITE;
			else
				exit(1);
			break;
		case 'b':
			dram_bitmask = strtol(optarg, NULL, 0);
			break;
		case 'B': /* set channel bitmask */
			channel_bitmask = strtol(optarg, NULL, 0);
			break;
       case 'P': /* select pseudo-channel (changed from priority) */
            g_pseudo_channel[g_pseudo_channel_cnt++] = strtol(optarg, NULL, 0);
            break;
		case 'c': /* set CPU affinity */
			cpuid = strtol(optarg, NULL, 0);
			fprintf(stderr, "cpuid: %d\n", cpuid);
			num_processors = sysconf(_SC_NPROCESSORS_CONF);
			CPU_ZERO(&cmask);
			CPU_SET(cpuid % num_processors, &cmask);
			if (sched_setaffinity(0, num_processors, &cmask) < 0) {
				perror("error");
				exit(1);
			}
			else
				fprintf(stderr, "assigned to cpu %d\n", cpuid);
			break;
		case 'C': /* select channel */
			g_channel[g_channel_cnt++] = strtol(optarg, NULL, 0);
			break;
		case 'd': /* debug */
			g_debug = strtol(optarg, NULL, 0);
			break;
		case 'e': /* select color (dram bank) */
			g_color[g_color_cnt++] = strtol(optarg, NULL, 0);
			break;	
		case 'p': /* set priority */
			prio = strtol(optarg, NULL, 0);
			if (setpriority(PRIO_PROCESS, 0, prio) < 0)
				perror("error");
			else
				fprintf(stderr, "assigned priority %d\n", prio);
			break;
		case 'i': /* iterations */
			repeat = strtol(optarg, NULL, 0);
			fprintf(stderr, "repeat=%ld\n", repeat);
			break;
		case 'l': /* MLP */
			mlp = strtol(optarg, NULL, 0);
			fprintf(stderr, "MLP=%d\n", mlp);
			break;
                case 'x':
			use_hugepage = (use_hugepage) ? 0: 1;
			break;
		case 's': /* synchronization ID */
			sync_id = strtol(optarg, NULL, 0);
			sync_enabled = 1;
			break;
		case 'n': /* number of attackers */
			num_attackers = strtol(optarg, NULL, 0);
			sync_enabled = 1;
			break;
		case 'A': /* is attacker */
			is_attacker = 1;
			break;
		case 'T': /* attack type */
			attack_type = strtol(optarg, NULL, 0);
			printf("Attack type: %d (0=pointer chasing, 1=matrix multiplication)\n", attack_type);
			break;
		case 'D': /* matrix dimension */
			matrix_dimension = strtol(optarg, NULL, 0);
			printf("Matrix dimension: %d\n", matrix_dimension);
			break;
		}

	}
	// After parsing options
	// Clean up any existing barrier files
	if (sync_enabled) {
		for (int i = 0; i <= num_attackers; i++) {
			char filename[100];
			sprintf(filename, "/tmp/barrier_%d", i);
			unlink(filename);  // Remove the file if it exists
		}
	}

	if (is_attacker) {
		signal(SIGTERM, signal_handler);
		signal(SIGINT, signal_handler);
		printf("CPU%d: Attacker ready to receive termination signal\n", cpuid);
	}
	///////////////////////////////////////////////////////////////////////

	// Allocate matrices if using matrix multiplication
	if ( attack_type == 1 || attack_type == 2) {
		int matrix_size = matrix_dimension * matrix_dimension * sizeof(float);
		matrix_A = (float*)malloc(matrix_size);
		matrix_B = (float*)malloc(matrix_size);
		matrix_C = (float*)malloc(matrix_size);
		
		if (!matrix_A || !matrix_B || !matrix_C) {
			printf("Failed to allocate matrices\n");
			exit(1);
		}
		
		init_matrix_data(matrix_A, matrix_B, matrix_C, matrix_dimension);
		printf("Matrices allocated and initialized (%dx%d, %.1f KB total)\n", 
			   matrix_dimension, matrix_dimension, 3.0 * matrix_size / 1024.0);
	}

	printf("sizeof(unsigned long): %d\n", (int)sizeof(unsigned long));
	printf("BITS_PER_LONG: %d\n", BITS_PER_LONG);
	unsigned long c;
	printf("DRAM Bank bits: ");
	for_each_set_bit(c, (&dram_bitmask), BITS_PER_LONG) {
		printf("%d ", (int)c);
	}
	printf("\n");
	if (g_color_cnt) {
		printf("Colors: ");
		for (int i = 0; i < g_color_cnt; i++) {
			printf("%d ", g_color[i]);
		}
		printf("\n");
	}

	printf("Channel bits: ");
	for_each_set_bit(c, (&channel_bitmask), BITS_PER_LONG) {
		printf("%d ", (int)c);
	}
	printf(" (mask: 0x%lx)\n", channel_bitmask);
	
	if (g_channel_cnt) {
		printf("Target Channels: ");
		for (int i = 0; i < g_channel_cnt; i++) {
			printf("%d ", g_channel[i]);
		}
		printf("\n");
	}

	printf("Pseudo-channel bits: ");
	for_each_set_bit(c, (&pseudo_channel_bitmask), BITS_PER_LONG) {
		printf("%d ", (int)c);
	}
	printf(" (mask: 0x%lx)\n", pseudo_channel_bitmask);

	if (g_pseudo_channel_cnt) {
		printf("Target Pseudo-channels: ");
		for (int i = 0; i < g_pseudo_channel_cnt; i++) {
			printf("%d ", g_pseudo_channel[i]);
		}
		printf("\n");
	}
	
	srand(0);

	int ws = 0;
	int orig_ws = (g_mem_size / CACHE_LINE_SIZE);

	printf("orig_ws: %d  mlp: %d\n", orig_ws, mlp);
	
	clock_gettime(CLOCK_REALTIME, &start);

	/* alloc memory. align to a page boundary */
	if (use_hugepage) {
		memchunk = (int *)mmap(0, 
				       g_mem_size,
				       PROT_READ | PROT_WRITE, 
				       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, 
				       -1, 0);
		if ((void *)memchunk == MAP_FAILED) {
			perror("alloc failed");
			exit(1);
		}
	} else {
		memchunk = (int *)malloc(g_mem_size);
		if (memchunk == NULL) {
			perror("alloc failed");
			exit(1);
		}
		printf("Using malloc(), not very accurate\n");
	}

	/* initialize data */
	memset(memchunk, 0, g_mem_size);

	if (attack_type == 0) {
		// set some values:
		for (int i=0; i<orig_ws; i++) {
			ulong vaddr = (ulong)&memchunk[i*CACHE_LINE_SIZE/4];
			bool should_add = false;

			if (g_color_cnt > 0 || g_channel_cnt > 0 || g_pseudo_channel_cnt > 0) {
				/* use coloring */
				bool bank_match = (g_color_cnt == 0); // default true if no bank filter
				bool channel_match = (g_channel_cnt == 0); // default true if no channel filter
				bool pseudo_channel_match = (g_pseudo_channel_cnt == 0); // default true if no pseudo-channel filter
					
				// Check bank match
				if (g_color_cnt > 0) {
					for (int j = 0; j < g_color_cnt; j++) {
						if (paddr_to_color(dram_bitmask, vaddr) == g_color[j]) {
							bank_match = true;
							break;
						}
					}
				}
					
				// Check channel match
				if (g_channel_cnt > 0) {
					for (int j = 0; j < g_channel_cnt; j++) {
						if (paddr_to_color(channel_bitmask, vaddr) == g_channel[j]) {
							channel_match = true;
							break;
						}
					}
				}
					
				if (g_pseudo_channel_cnt > 0) {
					for (int j = 0; j < g_pseudo_channel_cnt; j++) {
						if (paddr_to_color(pseudo_channel_bitmask, vaddr) == g_pseudo_channel[j]) {
							pseudo_channel_match = true;
							break;
						}
					}
				}

				should_add = bank_match && channel_match && pseudo_channel_match;
			
				if (should_add && g_debug) {
					printf("vaddr: %p bank: %d channel: %d pseudo: %d\n",
						(void *)vaddr,
						paddr_to_color(dram_bitmask, vaddr),
						paddr_to_color(channel_bitmask, vaddr),
						paddr_to_color(pseudo_channel_bitmask, vaddr));

				}
			} else {
				/* not using coloring or channel filtering */
				should_add = true;
			}
		
			if (should_add) {
				myvector.push_back(i);
			}
		}
	}
	// using built-in random generator:
	std::random_shuffle (myvector.begin(), myvector.end() );

	// update the workingset size
	ws = myvector.size() / mlp * mlp; 
	printf("new ws: %d\n", ws);
	int list_len = ws / mlp;
	printf("list_len: %d\n", list_len);
	
	for (i = 0; i < ws; i++) {
		int l = i / list_len;
		int curr_idx = myvector[i] * CACHE_LINE_SIZE / 4;
		int next_idx = myvector[i+1] * CACHE_LINE_SIZE / 4;
		if ((i+1) % list_len == 0)
			next_idx = myvector[i/list_len*list_len] * CACHE_LINE_SIZE / 4;
		
		memchunk[curr_idx] = next_idx;
		
		if (i % list_len == 0) {
			list[l] = memchunk;
			next[l] = curr_idx;
			printf("list[%d]  %d\n", l,  next[l] * 4 / CACHE_LINE_SIZE);
		}
		
		// printf("%8d ->%8d\n", myvector[i], next_idx*4/CACHE_LINE_SIZE);
	}
	
	if (use_hugepage) printf("Using hugetlb\n");

	clock_gettime(CLOCK_REALTIME, &end);
	printf("Init took %.0f us\n", (double) get_elapsed(&start, &end)/1000);

	long naccess = 0;

	if (sync_enabled) {
		sync_before_measurement();
	}

	clock_gettime(CLOCK_REALTIME, &start);

	/* actual access */
	if (attack_type == 1) {
		// Matrix multiplication with opt1
		printf("Running matrix multiplication attack with opt1\n");
		naccess = run_matrix_multiplication_opt1();
	} else if (attack_type == 2) {
		// Matrix multiplication with opt0
		printf("Running matrix multiplication attack with opt0\n");
		naccess = run_matrix_multiplication_opt0();
	} else {
		printf("Running pointer chasing attack\n");
		if (acc_type == READ)
			naccess = run((int64_t)repeat * list_len, mlp);
		else
			naccess = run_write((int64_t)repeat * list_len, mlp);
	}

	clock_gettime(CLOCK_REALTIME, &end);

	int64_t nsdiff = get_elapsed(&start, &end);
	double  avglat = (double)nsdiff/naccess;

	printf("alloc. size: %d (%d KB)\n", g_mem_size, g_mem_size/1024);
	int total_ws =  ws * CACHE_LINE_SIZE;
	printf("ws size: %d (%d KB)\n", total_ws, total_ws / 1024);
	printf("duration %.0f ns, #access %ld\n", (double)nsdiff, naccess);
	printf("Avg. latency %.2f ns\n", avglat);	
	printf("CPU%d: bandwidth %.2f MB/s\n", cpuid, (double)64*1000*naccess/nsdiff);

	// Print matrix-specific output if using matrix multiplication
	if (attack_type == 1 || attack_type == 2) {
		printf("Matrix dimension: %d\n", matrix_dimension);
		double checksum = matrix_checksum(matrix_C, matrix_dimension);
		printf("matmult_%s  secs: %.6f  chsum: %.6f\n", 
       			(attack_type == 1) ? "opt1" : "opt0", nsdiff / 1000000000.0, checksum);
	}

	// Clean up matrices if allocated
	if (attack_type == 1 || attack_type == 2) {
		free(matrix_A);
		free(matrix_B);
		free(matrix_C);
	}

	return 0;
}
