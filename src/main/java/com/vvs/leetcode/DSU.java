package com.vvs.leetcode;

import java.util.*;

public class DSU {

    private static int MAXX = (int) 1e5;
    private static int[] pre = new int[203030];
    private static List<Integer>[] lists = new ArrayList[MAXX + 10];

    static {
        for (int i = 0;i < lists.length; i++) {
            lists[i] = new ArrayList<>();
        }
        for (int i = 2;i <= MAXX; i++) {
            if (lists[i].isEmpty()) {
                for (int j = i;j <= MAXX; j += i) {
                    lists[j].add(i);
                }
            }
        }

    }

    public int find(int x) {
        if (pre[x] == x) {
            return pre[x];
        }
        return pre[x] = find(pre[x]);
    }

    public boolean canTraverseAllPairs(int[] nums) {
        int n = nums.length;
        int max = Arrays.stream(nums).max().getAsInt();
        for (int i = 0;i < max + n; i++) {
            pre[i] = i;
        }
        for (int i = 0;i < nums.length; i++) {
            for (int p : lists[nums[i]]) {
                int a = find(i), b = find(n + p);
                if (a == b) {
                    continue;
                }
                pre[a] = b;
            }
        }
        Set<Integer> set = new HashSet<>();
        for (int i = 0;i < nums.length; i++) {
            set.add(find(i));
        }
        return set.size() == 1;
    }

    public static void main(String[] args) {
    }
}
