package com.vvs.leetcode;

import java.util.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 23:24 2022/5/28
 * @Modified By:
 */
public class BookMyShow {

    int n, m;
    long[] trees;
    int[] mines;

    public BookMyShow(int n, int m) {
        this.n = n;
        this.m = m;
        trees = new long[n * 4];
        mines = new int[n * 4];
    }

    public void add(int k, int l, int r, int idx, int val) {
        if (l == r) {
            mines[k] += val;
            trees[k] += val;
            return ;
        }
        int mid = (l + r) >> 1;
        if (idx <= mid) {
            add(k * 2, l, mid, idx, val);
        } else {
            add(k * 2 + 1, mid + 1, r, idx, val);
        }
        mines[k] = Math.min(mines[k * 2], mines[k * 2 + 1]);
        trees[k] = trees[k * 2] + trees[k * 2 + 1];
    }

    public long sum(int k, int l, int r, int s, int e) {
        if (l >= s && r <= e) {
            return trees[k];
        }
        long sum = 0;
        int mid = (l + r) >> 1;
        if (s <= mid)
            sum += sum(k * 2, l, mid, s, e);
        if (e > mid) {
            sum += sum(k * 2 + 1, mid + 1, r, s, e);
        }
        return sum;
    }

    public int index(int k, int l, int r, int R, int val) {
        if (mines[k] > val) {
            return -1;
        }
        if (l == r) {
            return l;
        }
        int mid = (l + r) >> 1;
        if (mines[k * 2] <= val) {
            return index(k * 2, l, mid, R, val);
        }
        if (R > mid){
            return index(k * 2 + 1, mid + 1, r, R, val);
        }
        return -1;
    }

    public int[] gather(int k, int maxRow) {
        int vv = index(1, 1, n, maxRow + 1, m - k);
        if (vv == -1) {
            return new int[]{};
        }
        long cur = sum(1, 1, n, vv, vv);
        add(1, 1, n, vv, k);
        return new int[]{vv - 1, (int) cur};
    }

    public boolean scatter(int k, int maxRow) {
        if ((long) m * (maxRow + 1) - sum(1, 1, n, 1, maxRow + 1) < k) {
            return false;
        }
        for (int i = index(1, 1, n, maxRow + 1, m - 1);; i++) {
            int cur = (int) (m - sum(1, 1, n, i , i));
            if (k <= cur) {
                add(1, 1, n, i, k);
                return true;
            }
            k -= cur;
            add(1, 1, n, i, cur);
        }
    }

}
