#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define MAXQUEUE 100000

typedef struct {
    int vertex;
    int level;
} Node;

// Queue code roughly based on a project from CSC 230
// Visited needed to handle cross conflicts
// I also used ChatGPT to help debug some memory and related data structure problems
void bfs(int start, int n, int* row, int* col, int* visited) {
    int queueSize = 0;
    Node* queue = (Node*)malloc(MAXQUEUE*sizeof(Node));
    queue[queueSize++] = (Node){ start, 0 };
    visited[start] = 1;

    omp_set_num_threads(8);

    while (queueSize > 0) {
        #pragma omp parallel for
        for (int i = 0; i < queueSize; i++) {
            int vertex = queue[i].vertex;
            int level = queue[i].level;

            for (int j = row[vertex]; j < row[vertex + 1]; j++) {
                int neighbor = col[j];

                if (!visited[neighbor]) {
                    visited[neighbor] = 1;
                    queue[queueSize++] = (Node) { 
                        neighbor, 
                        level + 1
                    };
                }

            }
        }

        queueSize -= queueSize;
    }

    free(queue);
}

int main() {
    // Change this for number of nodes
    int n = 100000;
    // Change this for degree of each node
    int numEdges = 128;

    srand(12345);

    // In this code we reorganize a adjacency matrix into a set of arrays for potential speedup. This may also be friendlier
    // memory wise than creating the initial two-dimensional array. 
    int* row = (int*)malloc((n + 1)*sizeof(int));
    int* col = (int*)malloc(n*numEdges*sizeof(int));
    int* visited = (int*)calloc(n, sizeof(int));

    // I started running out of time so the createGraph method is inline in this version.
    row[0] = 0;
    for (int i = 0; i < n; i++) {
        int numNeighbors = 0;
        for (int j = 0; j < numEdges; j++) {
            int neighbor = rand() % n;
            if (neighbor != i) {
                col[row[i] + numNeighbors] = neighbor;
                numNeighbors++;
            }
        }
        row[i + 1] = row[i] + numNeighbors;
    }

    clock_t begin = clock();
    for (int w = 0; w < 200; w++) {
        bfs(0, n, row, col, visited);
    }
    clock_t end = clock();
    double timeSpent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Total time: %lf\n", timeSpent);
    printf("Time per instance: %lf\n", timeSpent/200);

    free(row);
    free(col);
    free(visited);

    return 0;
}
