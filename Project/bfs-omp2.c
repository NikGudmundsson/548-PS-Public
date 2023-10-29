#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MAXNODECOUNT 10000

// Referenced this for some help https://www.geeksforgeeks.org/introduction-and-array-implementation-of-queue/ and Dr. King's
// CSC 316 slideset for help on BFS and graph data structures
// I used ChatGPT to help debug some memory and related data structure problems

int queue[MAXNODECOUNT];
int visited[MAXNODECOUNT];
int adjacencyMatrix[MAXNODECOUNT][MAXNODECOUNT];

int front = -1, rear = -1;

void enqueue(int node) {
    if (rear == MAXNODECOUNT - 1) {
        printf("Queue overflow\n");
        exit(1);
    }

    if (front == -1) {
        front = 0;
    }

    rear++;
    queue[rear] = node;
}

int dequeue() {
    if (front == -1 || front > rear) {
        if (front == -1) {
            printf("\n-1\n");
        } 
        if (front > rear) {
            printf("\nf>r\n");
        }
        
        printf("Queue underflow\n");
        exit(1);
    }

    int node = queue[front];
    front++;

    return node;
}

void bfs(int start, int numNodes) {
    visited[start] = 1;
    enqueue(start);

    omp_set_num_threads(8);

    while (front != -1 && front <= rear) {
        int curr = dequeue();

        #pragma omp parallel for
        for (int i = 0; i < numNodes; i++) {
            if (adjacencyMatrix[curr][i] && !visited[i]) {
                visited[i] = 1;
                #pragma omp critical 
                {
                    enqueue(i);
                }
                
            }
        }
    }
}

int main() {
    int numNodes = 10000;

    srand(12345);

    for (int i = 0; i < numNodes; i++) {
        int count = 0;
        while (count < 8) {
            int j = rand() % numNodes;
            if (i != j && adjacencyMatrix[i][j] == 0) {
                adjacencyMatrix[i][j] = 1;
                adjacencyMatrix[j][i] = 1;
                count++;
            }
        }
    }

    int start = 1;

    printf("Breadth first traversal for a graph with %d nodes.\n", numNodes);
    clock_t begin = clock();

    bfs(start, numNodes);

    clock_t end = clock();

    double timeSpent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Total time: %lf\n", timeSpent);

    return 0;
}
