package com.vvs.leetcode;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:01 2022/4/29
 * @Modified By:
 */
public class ConstructQuadTree {

    public Node construct(int[][] grid) {
        return dfs(0, 0, grid.length, grid.length, grid);
    }

    public Node dfs(int x1, int y1, int x2, int y2, int[][] grid) {
        int cur = grid[x1][y1];
        boolean same = true;
        for (int i = x1;i <= x2; i++) {
            for (int j = y1;j <= y2; j++) {
                if (grid[i][j] != cur) {
                    same = false;
                    break;
                }
            }
        }
        if (same) {
            return new Node(cur == 1, true);
        }
        return new Node(
                false,
                false,
                dfs(x1, y1, (x1 + x2) / 2, (y1 + y2) / 2, grid),
                dfs(x1 , (y1 + y2) / 2, (x2 + x1) / 2, y2, grid),
                dfs((x1 + x2) / 2, y1, x2, (y1 + y2) / 2, grid),
                dfs((x1 + x2) / 2, (y1 + y2) / 2, x2, y2, grid)
        );
    }

    class Node {
        public boolean val;
        public boolean isLeaf;
        public Node topLeft;
        public Node topRight;
        public Node bottomLeft;
        public Node bottomRight;


        public Node() {
            this.val = false;
            this.isLeaf = false;
            this.topLeft = null;
            this.topRight = null;
            this.bottomLeft = null;
            this.bottomRight = null;
        }

        public Node(boolean val, boolean isLeaf) {
            this.val = val;
            this.isLeaf = isLeaf;
            this.topLeft = null;
            this.topRight = null;
            this.bottomLeft = null;
            this.bottomRight = null;
        }

        public Node(boolean val, boolean isLeaf, Node topLeft, Node topRight, Node bottomLeft, Node bottomRight) {
            this.val = val;
            this.isLeaf = isLeaf;
            this.topLeft = topLeft;
            this.topRight = topRight;
            this.bottomLeft = bottomLeft;
            this.bottomRight = bottomRight;
        }
    };
}
