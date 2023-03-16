package com.vvs.leetcode;

import java.util.HashSet;
import java.util.Set;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 10:43 2022/7/10
 * @Modified By:
 */
public class SmallestInfiniteSet {

    private Set<Integer> set;

    public SmallestInfiniteSet() {
        set = new HashSet<>();
    }

    public int popSmallest() {
        for (int i = 1;; i++) {
            if (!set.contains(i)) {
                set.add(i);
                return i;
            }
        }
    }

    public void addBack(int num) {
        if (set.contains(num)) {
            set.remove(num);
        }
    }
}
