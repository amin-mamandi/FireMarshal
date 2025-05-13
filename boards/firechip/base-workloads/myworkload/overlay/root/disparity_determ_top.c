#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <libgen.h>

#define MAP_OUTER_CACHE 0x200000  /* As defined in your kernel */
#define MAX_BUFSIZE (4 * 1024)     

int main(int argc, char *argv[]) {
    char *buf;
    int i;
    void *requested_addr = NULL;
    
    // Process the --ndmpgs argument
    for (i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--ndmpgs") == 0) {
            printf("Found --ndmpgs argument with value: %s\n", argv[i+1]);
        }
        else if (strcmp(argv[i], "--addr") == 0) {
            // Allow the user to specify a desired virtual address
            unsigned long addr = strtoul(argv[i+1], NULL, 16);
            if (addr != 0) {
                requested_addr = (void *)addr;
                printf("Requesting specific virtual address: %p\n", requested_addr);
            }
        }
    }
    
    printf("Attempting to allocate memory with MAP_OUTER_CACHE flag\n");
    
    /* Try to allocate memory with the MAP_OUTER_CACHE flag and at a specific address if requested */
    if (requested_addr != NULL) {
        buf = (char *)mmap(requested_addr, MAX_BUFSIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_OUTER_CACHE | MAP_FIXED, -1, 0);
    } else {
        buf = (char *)mmap(NULL, MAX_BUFSIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_OUTER_CACHE, -1, 0);
    }
    
    if (buf == MAP_FAILED) {
        printf("mmap with MAP_OUTER_CACHE failed: %s\n", strerror(errno));
        return 1;
    }
    
    printf("Memory allocated at address: %p\n", (void *)buf);
    
    /* Write a simple string to the allocated memory */
    snprintf(buf, MAX_BUFSIZE, "HelloFromAmin");
    printf("Content written to allocated memory: %s\n", buf);
    
    /* Access the memory to make sure it's properly allocated */
    for (i = 0; i < 12; i++) {
        printf("Memory byte %d: %c\n", i, buf[i]);
    }
    
    /* Properly unmap the memory when done */
    printf("Unmapping memory...\n");
    if (munmap(buf, MAX_BUFSIZE) == -1) {
        printf("munmap failed: %s\n", strerror(errno));
        return 1;
    }
    
    printf("Program completed successfully.\n");
    return 0;
}