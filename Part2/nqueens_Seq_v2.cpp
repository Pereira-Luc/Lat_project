/*
 * Author: Guillaume HELBECQUE (Université du Luxembourg)
 * Date: 10/10/2024
 *
 * Description:
 * This program solves the N-Queens problem using a sequential Depth-First tree-Search
 * (DFS) algorithm. It serves as a basis for task-parallel implementations.
 */

#include <iostream>
#include <cstring>
#include <utility>
#include <vector>
#include <chrono>
#include <stack>

// Static functions for inequalities
static bool no_same_column(const std::vector<int> &board, int row, int col)
{
  for (int i = 0; i < row; ++i)
  {
    if (board[i] == col)
    {
      return false;
    }
  }
  return true;
}

static bool no_same_diagonal(const std::vector<int> &board, int row, int col)
{
  for (int i = 0; i < row; ++i)
  {
    if (board[i] == col - row + i || board[i] == col + row - i)
    {
      return false;
    }
  }
  return true;
}

static bool no_column_zero_if_even_row(const std::vector<int> &board, int row, int col)
{
  return !(row % 2 == 0 && col == 0);
}

// N-Queens node
struct Node
{
  int depth;                            // Depth in the tree
  std::vector<int> board;               // Board configuration (permutation)
  std::vector<std::vector<bool> > domain; // Domain for each row

  Node(size_t N) : depth(0), board(N), domain(N, std::vector<bool>(N, true))
  {
    for (int i = 0; i < N; i++)
    {
      board[i] = i; // Initialize board with default column indices
    }
  }
  Node(const Node &) = default;
  Node(Node &&) = default;
  Node() = default;
};

// Apply arbitrary inequalities
bool check_inequalities(const std::vector<int> &board, int row, int col,
                        const std::vector<bool (*)(const std::vector<int> &, int, int)> &inequalities)
{
  for (const auto &inequality : inequalities)
  {
    if (!inequality(board, row, col))
    {
      return false;
    }
  }
  return true;
}

// Function to propagate domain reduction until a fixpoint is reached
bool propagate_domains(Node &node, size_t N)
{
    bool updated;
    do
    {
        updated = false;

        for (int row = 0; row < node.depth; ++row)
        {
            int col = node.board[row]; // Column where the queen is placed in `row`

            // Propagate domain reduction to other rows
            for (int i = row + 1; i < N; ++i)
            {
                // If the value is still in the domain, remove it and mark as updated
                if (node.domain[i][col])
                {
                    node.domain[i][col] = false;
                    updated = true;
                }

                // Remove diagonals
                if (col - (i - row) >= 0 && node.domain[i][col - (i - row)])
                {
                    node.domain[i][col - (i - row)] = false;
                    updated = true;
                }

                if (col + (i - row) < N && node.domain[i][col + (i - row)])
                {
                    node.domain[i][col + (i - row)] = false;
                    updated = true;
                }
            }
        }

    } while (updated); // Continue until no updates are made

    // Check if any row has an empty domain; this means the node is invalid
    for (int row = node.depth; row < N; ++row)
    {
        bool hasValidValue = false;
        for (int col = 0; col < N; ++col)
        {
            if (node.domain[row][col])
            {
                hasValidValue = true;
                break;
            }
        }
        if (!hasValidValue)
        {
            return false; // Dead-end: No valid placements possible for this node
        }
    }

    return true; // Valid node
}

// Evaluate and branch function with fixpoint domain propagation
void evaluate_and_branch(const Node &parent, std::stack<Node> &pool, size_t &tree_loc, size_t &num_sol,
                         const std::vector<bool (*)(const std::vector<int> &, int, int)> &inequalities)
{
    int depth = parent.depth;
    int N = parent.board.size();

    // If the node is a leaf, count it as a solution
    if (depth == N)
    {
        num_sol++;
        return;
    }

    // Iterate over the domain of the current row
    for (int col = 0; col < N; ++col)
    {
        if (parent.domain[depth][col]) // Check if column is in the domain
        {
            Node child(parent);
            child.board[depth] = col; // Place the queen
            child.depth++;

            // Reduce domain for the child and propagate fixpoint
            if (propagate_domains(child, N))
            {
                pool.push(std::move(child));
                tree_loc++;
            }
        }
    }
}

int main(int argc, char **argv)
{
  // helper
  if (argc != 2)
  {
    std::cout << "usage: " << argv[0] << " <number of queens> " << std::endl;
    exit(1);
  }

  std::vector<bool (*)(const std::vector<int> &, int, int)> inequalities;
  inequalities.push_back(no_same_column);
  inequalities.push_back(no_same_diagonal);
  // inequalities.push_back(no_column_zero_if_even_row);

  // problem size (number of queens)
  size_t N = std::stoll(argv[1]);
  std::cout << "Solving " << N << "-Queens problem\n"
            << std::endl;

  // initialization of the root node (the board configuration where no queen is placed)
  Node root(N);

  // initialization of the pool of nodes (stack -> DFS exploration order)
  std::stack<Node> pool;
  pool.push(std::move(root));

  // statistics to check correctness (number of nodes explored and number of solutions found)
  size_t exploredTree = 0;
  size_t exploredSol = 0;

  // beginning of the Depth-First tree-Search
  auto start = std::chrono::steady_clock::now();

  while (pool.size() != 0)
  {
    // get a node from the pool
    Node currentNode(std::move(pool.top()));
    pool.pop();

    // check the board configuration of the node and branch it if it is valid.
    evaluate_and_branch(currentNode, pool, exploredTree, exploredSol, inequalities);
  }

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // outputs
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
  std::cout << "Total solutions: " << exploredSol << std::endl;
  std::cout << "Size of the explored tree: " << exploredTree << std::endl;

  return 0;
}
