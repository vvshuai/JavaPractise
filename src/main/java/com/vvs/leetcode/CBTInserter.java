package com.vvs.leetcode;

import java.util.ArrayDeque;
import java.util.Queue;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:15 2022/7/25
 * @Modified By:
 */
public class CBTInserter {

    TreeNode root;
    Queue<TreeNode> que;

    public CBTInserter(TreeNode root) {
        this.root = root;
        que = new ArrayDeque<>();
        Queue<TreeNode> cur = new ArrayDeque<>();
        if (root != null) {
            cur.add(root);
            que.add(root);
            while (!cur.isEmpty()) {
                for (int i = cur.size() - 1;i >= 0; i--) {
                    TreeNode curRoot = cur.poll();
                    if (curRoot.left != null) {
                        cur.add(curRoot.left);
                        que.add(curRoot.left);
                    }
                    if (curRoot.right != null) {
                        cur.add(curRoot.right);
                        que.add(curRoot.right);
                    }
                }
            }
        }
    }

    public int insert(int val) {
        for (TreeNode cur : que) {
            if (cur.left == null) {
                cur.left = new TreeNode(val);
                que.add(cur.left);
                return cur.val;
            } else if (cur.right == null) {
                cur.right = new TreeNode(val);
                que.add(cur.right);
                return cur.val;
            }
        }
        return 0;
    }

    public TreeNode get_root() {
        return root;
    }

}
