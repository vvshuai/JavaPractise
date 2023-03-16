package com.vvs.leetcode;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:24 2022/7/24
 * @Modified By: 区间线段树， 动态开点
 */
public class SegmentTreeDynamic {
    static class Node {
        Node left, right;
        int val;
        int add;
    }
    private final int N = (int) 1e9;
    private final Node root = new Node();

    public void update(Node node, int start, int end, int l, int r, int val) {
        if (l <= start && end <= r) {
            node.val += val;
            node.add += val;
            return ;
        }
        int mid = (start + end) >> 1;
        pushDown(node);
        if (l <= mid) {
            update(node.left, start, mid, l, r, val);
        }
        if (r > mid) {
            update(node.right, mid + 1, end, l, r, val);
        }
        node.val = Math.max(node.left.val, node.right.val);
    }

    public int query(Node node, int start, int end, int l, int r) {
        if (l <= start && end <= r) {
            return node.val;
        }
        int ans = 0;
        int mid = (start + end) >> 1;
        // 有可能查的区间不存在，所以需要pushDown
        pushDown(node);
        if (l <= mid) {
            ans = query(node.left, start, mid, l, r);
        }
        if (r > mid) {
            ans = Math.max(ans, query(node.right, mid + 1, end, l, r));
        }
        return ans;
    }

    private void pushDown(Node node) {
        if (node.left == null) {
            node.left = new Node();
        }
        if (node.right == null) {
            node.right = new Node();
        }
        // 没有需要的累加值
        if (node.add == 0) {
            return ;
        }
        // 将标记下移到Node的左右区间内，并将改区间清空
        node.left.val += node.add;
        node.right.val +=  node.add;
        node.left.add += node.add;
        node.right.add += node.add;
        node.add = 0;
    }

    public boolean book(int start, int end) {
        if (query(root, 0, N, start, end) == 1) {
            return false;
        }
        update(root, 0, N, start, end, 1);
        return true;
    }
}
