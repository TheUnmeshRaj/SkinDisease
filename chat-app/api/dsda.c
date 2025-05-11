#include <stdio.h>
#include <stdlib.h>

// Define a node for the BST
typedef struct Node {
    int speed;
    struct Node *left, *right;
} Node;

// Function to create a new node
Node* createNode(int speed) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->speed = speed;
    newNode->left = NULL;
    newNode->right = NULL;
    return newNode;
}

// Insert a new speed into the BST
Node* insertNode(Node* root, int speed) {
    if (root == NULL) return createNode(speed);
    if (speed < root->speed)
        root->left = insertNode(root->left, speed);
    else
        root->right = insertNode(root->right, speed);
    return root;
}

// Preorder traversal: Root -> Left -> Right
void preorder(Node* root) {
    if (root == NULL) return;
    printf("%d ", root->speed);
    preorder(root->left);
    preorder(root->right);
}

// Inorder traversal: Left -> Root -> Right
void inorder(Node* root) {
    if (root == NULL) return;
    inorder(root->left);
    printf("%d ", root->speed);
    inorder(root->right);
}

// Postorder traversal: Left -> Right -> Root
void postorder(Node* root) {
    if (root == NULL) return;
    postorder(root->left);
    postorder(root->right);
    printf("%d ", root->speed);
}

// Sort and Display the Speeds
void sortAndDisplaySpeeds(int speeds[], int n) {
    printf("Vehicle Speeds (Unsorted):\n");
    for (int i = 0; i < n; i++)
        printf("%d ", speeds[i]);
    printf("\n");

    // Sort the speeds in increasing order
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (speeds[j] > speeds[j + 1]) {
                int temp = speeds[j];
                speeds[j] = speeds[j + 1];
                speeds[j + 1] = temp;
            }
        }
    }

    printf("Vehicle Speeds (Sorted in Increasing Order):\n");
    for (int i = 0; i < n; i++)
        printf("%d ", speeds[i]);
    printf("\n");
}

// Build BST from speeds
Node* buildBST(int speeds[], int n) {
    Node* root = NULL;
    for (int i = 0; i < n; i++)
        root = insertNode(root, speeds[i]);
    return root;
}

int main() {
    // Define vehicle speeds
    int speeds[] = {45, 60, 30, 75, 55, 90, 50};
    int n = sizeof(speeds) / sizeof(speeds[0]);

    // Sort and display speeds
    sortAndDisplaySpeeds(speeds, n);

    // Build a binary search tree from the speeds
    Node* bstRoot = buildBST(speeds, n);

    // Display traversals
    printf("\nPreorder Traversal (Root -> Left -> Right):\n");
    preorder(bstRoot);

    printf("\n\nInorder Traversal (Left -> Root -> Right):\n");
    inorder(bstRoot);

    printf("\n\nPostorder Traversal (Left -> Right -> Root):\n");
    postorder(bstRoot);

    return 0;
}
