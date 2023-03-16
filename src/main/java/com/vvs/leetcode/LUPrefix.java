package com.vvs.leetcode;

import java.util.TreeSet;

public class LUPrefix {

    int[] arr;
    TreeSet<Integer> set = new TreeSet<>();

    public LUPrefix(int n) {
        arr = new int[n];
        for (int i = 1;i <= n; i++) {
            set.add(i);
        }
    }

    public void upload(int video) {
        set.remove(video);
    }

    public int longest() {
        Integer first = set.first();
        return first - 1;
    }
}
