#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>

#define MAX_NODES 100000

// I used ChatGPT to help debug some memory and related data structure problems
// Some implementation built up from an earlier BFS. This organization of nodes felt better because they contained more
// information.

typedef struct node {
    int id;
    int distance;
    bool visited;
    struct node** neighbors;
    int numNeighbors;
} Node;

typedef struct graph {
    Node* nodes[MAX_NODES];
    int numNodes;
} Graph;

Graph* create_graph(int numNodes) {
    Graph* g = malloc(sizeof(Graph));
    g->numNodes = numNodes;

    for (int i = 0; i < numNodes; i++) {
        Node* node = malloc(sizeof(Node));
        node->id = i;
        node->distance = INT_MAX;
        node->visited = false;
        // Change number here to change degree
        node->neighbors = malloc(128*sizeof(Node*));
        node->numNeighbors = 0;

        g->nodes[i] = node;
    }

    for (int i = 0; i < numNodes; i++) {
        Node* node = g->nodes[i];
        // Change upper bound to change degree. Make sure is same as number above.
        for (int j = 0; j < 128; j++) {
            int neighborID = rand() % numNodes;
            while (neighborID == i) {
                neighborID = rand() % numNodes;
            }
            Node* neighbor = g->nodes[neighborID];
            node->neighbors[node->numNeighbors++] = neighbor;
        }
    }

    return g;
}

void bfs(Graph* g, Node* start) {
    start->distance = 0;
    start->visited = true;

    Node* queue[MAX_NODES];
    int front = 0;
    int rear = 0;
    queue[rear++] = start;

    while (front != rear) {
        Node* current = queue[front++];

        for (int i = 0; i < current->numNeighbors; i++) {
            Node* neighbor = current->neighbors[i];
            if (!neighbor->visited) {
                neighbor->visited = true;
                neighbor->distance = current->distance + 1;
                queue[rear++] = neighbor;
            }
        }
    }
}

int main() {
    Graph* g = create_graph(100000);
    Node* start = g->nodes[0];

    clock_t begin = clock();
    for (int w = 0; w < 200; w++) {
        bfs(g, start);
    }
    clock_t end = clock();
    double timeSpent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Total time: %lf\n", timeSpent);
    printf("Time per instance: %lf\n", timeSpent/200);

    return 0;
}
