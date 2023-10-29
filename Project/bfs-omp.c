#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MAXNODECOUNT 1000000

// Original implementation, I did not manage node creation very well and I think it was causing allocation issues.
// There were also a number of additional limitations that I had to completely change for bfs-omp3. 

// I referenced work from https://www.geeksforgeeks.org/introduction-and-array-implementation-of-queue/, a CSC 230 project, 
// and Dr. King's CSC 316 slide set on graph data structures and BFS
// I used ChatGPT to help debug some memory and related data structure problems

typedef struct node {
    int dest;
    struct node* next;
} Node;

typedef struct graph {
    int numNodes;
    Node* adjacencyList[MAXNODECOUNT];
} Graph;

Node* create_node(int dest) {
    Node* new = (Node*)malloc(sizeof(Node));
    new->dest = dest;
    new->next = NULL;
    return new;
}

void add_edge(Graph* g, int src, int dest) {
    Node* new = create_node(dest);
    new->next = g->adjacencyList[src];
    g->adjacencyList[src] = new;
}

Graph* createGraph(int numNodes) {
    Graph* g = (Graph*)malloc(sizeof(Graph));
    g->numNodes = numNodes;

    for (int i = 0; i < numNodes; i++) {
        g->adjacencyList[i] = NULL;
    }

    srand(12345);

    for (int i = 0; i < numNodes; i++) {
        int neighbor1 = rand() % numNodes;
        int neighbor2 = rand() % numNodes;

        add_edge(g, i, neighbor1);
        add_edge(g, i, neighbor2);
    }

    return g;
}

void bfs(Graph* g, int start, int* visited) {
    for (int i = 0; i < g->numNodes; i++) {
        visited[i] = 0;
    }

    int queue[MAXNODECOUNT];
    int head = 0, tail = 0;
    queue[tail++] = start;
    visited[start] = 1;

    while (head != tail) {
        int curr_node = queue[head++];

        omp_set_num_threads(8);

        #pragma omp parallel for
        for (int i = 0; i < g->numNodes; i++) {
            if (visited[i] == 0) {

                Node* neighborNode = g->adjacencyList[curr_node];

                while (neighborNode != NULL) {
                    if (neighborNode->dest == i) {
                        visited[i] = 1;
                        queue[tail++] = i;
                        break;
                    }
                    neighborNode = neighborNode->next;
                }
            }
        }
    }
}

int main() {
    Graph* g = createGraph(10000);
    int visited[10000];

    clock_t begin = clock();
    for (int w = 0; w < 200; w++) {
        bfs(g, 0, visited);
    }
    clock_t end = clock();

    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Total time: %lf\n", time_spent);
    printf("Time per instance: %lf\n", time_spent/200);

    for (int i = 0; i < g->numNodes; i++) {
        Node* neighborNode = g->adjacencyList[i];
        while (neighborNode != NULL) {
            Node* temp = neighborNode;
            neighborNode = neighborNode->next;
            free(temp);
        }
    }

    free(g);

    return 0;
}