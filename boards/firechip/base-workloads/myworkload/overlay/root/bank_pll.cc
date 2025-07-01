//////////////////////////////////////////////////////////////////////////////////////////////////////////

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

/**************************************************************************
 * Included Files
 **************************************************************************/
#include <iostream>     // std::cout
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <signal.h>

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
#include <m5ops.h>

///////////////////////////////////////////////////////////////////////////

// Add these to your includes
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <signal.h>

// Add these global variables
static int num_attackers = 0;
static int is_attacker = 0;
static int sync_enabled = 0;
static int sync_id = -1;
static int sem_id = -1;
static int streaming_mode = 0;

static volatile int keep_running = 1;
static uint64_t actual_start_ticks = 0;
static int64_t actual_accesses = 0;

// Semaphore union definition required for some systems
union semun {
    int val;
    struct semid_ds *buf;
    unsigned short *array;
};

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
            // printf("Process %d: Waiting... (found %d/%d processes)\n", sync_id, count, num_attackers + 1);
            usleep(10000); // 10ms delay to avoid tight polling
        }
        attempt++;
    }
    
    printf("Process %d: All processes ready (%d/%d)\n", sync_id, count, num_attackers + 1);
    
    // If we're the victim process, take checkpoint and reset stats
    if (!is_attacker) {
        printf("All processes ready, taking checkpoint...\n");
        m5_checkpoint(0, 0);
        printf("Checkpoint taken, waiting 50ms before continuing...\n");
        
        // Reset stats only for the victim
        // printf("Victim resetting stats\n");
        // m5_reset_stats(0, 0);
        
        // Give attackers time to warm up caches
        usleep(50000); // 100ms delay
        printf("Victim: Continuing after warmup period\n");
    } else {
        // Attackers continue immediately after all processes are ready
        printf("Process %d: Attacker continuing\n", sync_id);
    }
}

///////////////////////////////////////////////////////////////////////////
/**************************************************************************
 * Public Definitions
 **************************************************************************/
#define MAX_MLP 32
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

// static unsigned long dram_bitmask = 0x1e000; // 16|15,14,13,--| : xu4 (cortex-a15)
// static unsigned long dram_bitmask = 0x7800;  // --,14,13,12|11  : pi4 (cortex-a72)
static unsigned long dram_bitmask = 0x78000;  // Bank bits [15:18] for 8 channels

static unsigned long channel_bitmask = 0x380; // Channel bits [7:9] for 8 channels
static unsigned long pseudo_channel_bitmask = 0x40; // Pseudo-channel bit [6] for HBM

static int g_channel[MAX_COLORS]; // Target channels
static int g_channel_cnt = 0;

static int g_pseudo_channel[MAX_COLORS]; // Target pseudo-channels
static int g_pseudo_channel_cnt = 0;

int *g_mem_ptr = 0;           // pointer to allocated memory region  
int *memchunk = 0;

volatile uint64_t g_nread = 0; // number of bytes read

static int attack_type = 0;  // 0=pointer chasing, 1=matrix multiplication
static float *matrix_A = NULL;
static float *matrix_B = NULL;
static float *matrix_C = NULL;
static int matrix_dimension = 512;  // Default matrix size

typedef unsigned long ulong;

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

void print_final_bandwidth(int cpuid, uint64_t start_time, uint64_t end_time, uint64_t accesses, int access_type, int attack_type) {
    uint64_t elapsed_ns = end_time - start_time;
    double elapsed_s = (double)elapsed_ns / 1000000000.0;
    
    uint64_t total_bytes;
    if (attack_type == 1) {
        // For matrix multiplication, use the accesses as approximation
        total_bytes = accesses * sizeof(float) * 3; // Rough estimate for A, B, C access
    } else {
        total_bytes = accesses * 64; // 64 bytes per cache line for pointer chasing
    }
    
    double bandwidth_MBs = (double)total_bytes / elapsed_s / 1024.0 / 1024.0;
    
    printf("CPU%d: FINAL REPORT: %.2f MB/s (%.6f s, %lu accesses, %s, type=%d)\n", 
           cpuid, bandwidth_MBs, elapsed_s, accesses,
           (access_type == READ) ? "READ" : "WRITE", attack_type);
    fflush(stdout);
}

/**************************************************************************
 * Implementation
 **************************************************************************/
int64_t run(int64_t iter, int mlp)
{
    int64_t cnt = 0;
    actual_start_ticks = m5_rpns();

    for (int64_t i = 0; i < iter && keep_running; i++) {
        switch (mlp) {
        case 32:
            next[31] = list[31][next[31]];
        case 31:
            next[30] = list[30][next[30]];
        case 30:
            next[29] = list[29][next[29]];
        case 29:
            next[28] = list[28][next[28]];
        case 28:
            next[27] = list[27][next[27]];
        case 27:
            next[26] = list[26][next[26]];
        case 26:
            next[25] = list[25][next[25]];
        case 25:
            next[24] = list[24][next[24]];
        case 24:
            next[23] = list[23][next[23]];
        case 23:
            next[22] = list[22][next[22]];
        case 22:
            next[21] = list[21][next[21]];
        case 21:
            next[20] = list[20][next[20]];
        case 20:
            next[19] = list[19][next[19]];
        case 19:
            next[18] = list[18][next[18]];
        case 18:
            next[17] = list[17][next[17]];
        case 17:
            next[16] = list[16][next[16]];
        case 16:
            next[15] = list[15][next[15]];
        case 15:
            next[14] = list[14][next[14]];
        case 14:
            next[13] = list[13][next[13]];
        case 13:
            next[12] = list[12][next[12]];
        case 12:
            next[11] = list[11][next[11]];
        case 11:
            next[10] = list[10][next[10]];
        case 10:
            next[9] = list[9][next[9]];
        case 9:
            next[8] = list[8][next[8]];
        case 8:
            next[7] = list[7][next[7]];
        case 7:
            next[6] = list[6][next[6]];
        case 6:
            next[5] = list[5][next[5]];
        case 5:
            next[4] = list[4][next[4]];
        case 4:
            next[3] = list[3][next[3]];
        case 3:
            next[2] = list[2][next[2]];
        case 2:
            next[1] = list[1][next[1]];
        case 1:
            next[0] = list[0][next[0]];
        }
        cnt += mlp;
    }
    
    actual_accesses = cnt;
    return cnt;
}

long run_write(long iter, int mlp)
{
    long cnt = 0;
    actual_start_ticks = m5_rpns();

    for (long i = 0; i < iter && keep_running; i++) {
        switch (mlp) {
        case 32:
            list[31][next[31]+1] = 0xff;
            next[31] = list[31][next[31]];
        case 31:
            list[30][next[30]+1] = 0xff;
            next[30] = list[30][next[30]];
        case 30:
            list[29][next[29]+1] = 0xff;
            next[29] = list[29][next[29]];
        case 29:
            list[28][next[28]+1] = 0xff;
            next[28] = list[28][next[28]];
        case 28:
            list[27][next[27]+1] = 0xff;
            next[27] = list[27][next[27]];
        case 27:
            list[26][next[26]+1] = 0xff;
            next[26] = list[26][next[26]];
        case 26:
            list[25][next[25]+1] = 0xff;
            next[25] = list[25][next[25]];
        case 25:
            list[24][next[24]+1] = 0xff;
            next[24] = list[24][next[24]];
        case 24:
            list[23][next[23]+1] = 0xff;
            next[23] = list[23][next[23]];
        case 23:
            list[22][next[22]+1] = 0xff;
            next[22] = list[22][next[22]];
        case 22:
            list[21][next[21]+1] = 0xff;
            next[21] = list[21][next[21]];
        case 21:
            list[20][next[20]+1] = 0xff;
            next[20] = list[20][next[20]];
        case 20:
            list[19][next[19]+1] = 0xff;
            next[19] = list[19][next[19]];
        case 19:
            list[18][next[18]+1] = 0xff;
            next[18] = list[18][next[18]];
        case 18:
            list[17][next[17]+1] = 0xff;
            next[17] = list[17][next[17]];
        case 17:
            list[16][next[16]+1] = 0xff;
            next[16] = list[16][next[16]];
        case 16:
            list[15][next[15]+1] = 0xff;
            next[15] = list[15][next[15]];
        case 15:
            list[14][next[14]+1] = 0xff;
            next[14] = list[14][next[14]];
        case 14:
            list[13][next[13]+1] = 0xff;
            next[13] = list[13][next[13]];
        case 13:
            list[12][next[12]+1] = 0xff;
            next[12] = list[12][next[12]];
        case 12:
            list[11][next[11]+1] = 0xff;
            next[11] = list[11][next[11]];
        case 11:
            list[10][next[10]+1] = 0xff;
            next[10] = list[10][next[10]];
        case 10:
            list[9][next[9]+1] = 0xff;
            next[9] = list[9][next[9]];
        case 9:
            list[8][next[8]+1] = 0xff;
            next[8] = list[8][next[8]];
        case 8:
            list[7][next[7]+1] = 0xff;
            next[7] = list[7][next[7]];
        case 7:
            list[6][next[6]+1] = 0xff;
            next[6] = list[6][next[6]];
        case 6:
            list[5][next[5]+1] = 0xff;
            next[5] = list[5][next[5]];
        case 5:
            list[4][next[4]+1] = 0xff;
            next[4] = list[4][next[4]];
        case 4:
            list[3][next[3]+1] = 0xff;
            next[3] = list[3][next[3]];
        case 3:
            list[2][next[2]+1] = 0xff;
            next[2] = list[2][next[2]];
        case 2:
            list[1][next[1]+1] = 0xff;
            next[1] = list[1][next[1]];
        case 1:
            list[0][next[0]+1] = 0xff;
            next[0] = list[0][next[0]];
        }
        cnt += mlp;
    }
    
    actual_accesses = cnt;
    return cnt;
}

int64_t run_stream_read(int ws, std::vector<int> &myvector) {
    int i;    
    int64_t sum = 0;
    
    // Use the same working set that pointer chasing uses
    for (i = 0; i < ws; i++) {
        int cache_line_idx = myvector[i] * CACHE_LINE_SIZE / 4;
        sum += memchunk[cache_line_idx];
    }
    
    g_nread += ws * CACHE_LINE_SIZE;
    return sum;
}

int64_t run_stream_write(int ws, std::vector<int> &myvector) {
    int i;    
    
    // Use the same working set that pointer chasing uses
    for (i = 0; i < ws; i++) {
        int cache_line_idx = myvector[i] * CACHE_LINE_SIZE / 4;
        memchunk[cache_line_idx] = i;
    }
    
    g_nread += ws * CACHE_LINE_SIZE;
    return 1;
}


int main(int argc, char* argv[])
{
	// struct sched_param param;
        cpu_set_t cmask;
	int num_processors;
	int cpuid = 0;

	int opt, prio;
	int i;

	long repeat = DEFAULT_ITER;
	int mlp = 1;
	int use_hugepage = 0;
	struct timespec start, end;
	uint64_t  start_ticks, end_ticks;
	int acc_type = READ;

	std::srand (0);
	std::vector<int> myvector;

	/*
	 * get command line options 
	 */
	while ((opt = getopt(argc, argv, "m:a:c:d:e:C:b:B:P:i:l:hxs:n:AS:T:D")) != -1) {
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
		case 'S': /* streaming mode */
    		streaming_mode = 1;
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
	///////////////////////////////////////////////////////////////////////
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
	// start_ticks = m5_rpns();

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
	memset((char *)memchunk, 1, g_mem_size);

	for (i = 0; i < g_mem_size / sizeof(int); i++)
		memchunk[i] = i;

	// set some values:
	for (int i=0; i<orig_ws; i++) {
		ulong vaddr = (ulong)&memchunk[i*CACHE_LINE_SIZE/4];
		bool should_add = false;

		if (g_color_cnt > 0 || g_channel_cnt > 0 || g_pseudo_channel_cnt > 0) {
			/* use coloring and/or channel filtering */
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

#if 0
        param.sched_priority = 1;
        if(sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
		perror("sched_setscheduler failed");
        }
#endif

	clock_gettime(CLOCK_REALTIME, &end);
	// end_ticks = m5_rpns();
	printf("Init took %lu ticks \n", (end_ticks - start_ticks));
	printf("Init took %.0f us\n", (double) get_elapsed(&start, &end)/1000);

	long naccess;
	long sum = 0;
	unsigned int g_start, g_end;
	
	/////////////////////////////////////////////////////////////////////////////////////
	// Right before measurement - after initialization but before the measurement code
	if (sync_enabled) {
		sync_before_measurement();
	}
	/////////////////////////////////////////////////////////////////////////////////////

	// Use bandwidth benchmark style timing for streaming mode
	if (streaming_mode) {
		printf("Starting streaming benchmark...\n");
		start_ticks = m5_rpns();
		printf("start ticks: %lu\n", start_ticks);
		
		for (i=0;; i++) {
			switch (acc_type) {
			case READ:
				sum += run_stream_read(ws, myvector);
				break;
			case WRITE:
				sum += run_stream_write(ws, myvector);
				break;
			}
			if (repeat > 0 && i+1 >= repeat)
				break;
		}
			
		end_ticks = m5_rpns();
		printf("end ticks: %lu\n", end_ticks);

		printf("total sum = %ld\n", (long)sum);

		uint64_t elapsed_ticks = end_ticks - start_ticks;
		printf("elapsed ticks (ns): %lu\n", elapsed_ticks);

		// Fixed time conversion: ns to seconds
		float dur_in_sec = (double)elapsed_ticks / 1000000000.0; // ns to seconds
		printf("elapsed s: %.6f\n", dur_in_sec);
		
		// Calculate and print bandwidth benchmark style results
		printf("total sum = %ld\n", (long)sum);
		float bw;
		printf("g_nread(bytes read) = %lld\n", (long long)g_nread);

		bw = (float)g_nread / dur_in_sec / 1024 / 1024;
		printf("CPU%d: B/W = %.2f MB/s | ", cpuid, bw);
		printf("CPU%d: average = %.2f ns\n", cpuid, (dur_in_sec*1000000000)/(g_nread/CACHE_LINE_SIZE));
		m5_dump_reset_stats(0, 0);
		
	} else {
		// Choose attack type
		start_ticks = m5_rpns();
		printf("start ticks: %lu\n", start_ticks);
		
		// Original pointer chasing attack
		printf("Running pointer chasing attack\n");
		if (acc_type == READ)
			naccess = run(repeat * list_len, mlp);
		else
			naccess = run_write(repeat * list_len, mlp);
			
		end_ticks = m5_rpns();
		printf("end ticks: %lu\n", end_ticks);

		if (is_attacker && !keep_running) {
			// Attacker was terminated, use actual measured time
			end_ticks = m5_rpns();
			naccess = actual_accesses;
			printf("CPU%d: Attacker terminated after %.6f seconds\n", cpuid, 
				(double)(end_ticks - actual_start_ticks) / 1000000000.0);
    	}
		
		uint64_t elapsed_ticks = end_ticks - start_ticks;
		// printf("elapsed (ns): %lu\n", elapsed_ticks);

		double elapsed_s = (double)elapsed_ticks / 1000000000.0;
		printf("elapsed (s): %.6f\n", elapsed_s);

		uint64_t total_bytes;

		total_bytes = naccess * 64;


		double bandwidth_MBs = (double)total_bytes / elapsed_s / 1024.0 / 1024.0;
		double bandwidth_mb = (double)total_bytes / ((double)elapsed_ticks) * 1000;

		printf("naccess: %ld\n", naccess);
		printf("total bytes: %lu\n", total_bytes);
		printf("bandwidth %.2f MB/s\n", bandwidth_MBs);


		if (attack_type == 1) {
			print_final_bandwidth(cpuid, start_ticks, end_ticks, naccess, READ, attack_type);
		} else {
			print_final_bandwidth(cpuid, start_ticks, end_ticks, naccess, acc_type, attack_type);
		}
		m5_dump_reset_stats(0, 0);
	}
}