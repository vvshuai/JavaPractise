package com.vvs.leetcode;

import java.util.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 11:17 2022/5/15
 * @Modified By:
 */
public class CountIntervals {

    int ans = 0;
    TreeSet<int[]> set = new TreeSet<>(Comparator.comparingInt(o -> o[0]));

    public CountIntervals() {

    }

    public void add(int left, int right) {
        while (!set.isEmpty()) {
            int[] tem = set.ceiling(new int[]{left - 1, -1});
            if (tem == null || tem[0] > right + 1) {
                break;
            }
            // 合并区间
            int[] cur = new int[] {Math.min(left, tem[0]), Math.max(right, tem[1])};
            left = cur[0];
            right = cur[1];
            set.remove(tem);
            ans -= (tem[1] - tem[0] + 1);
        }
        ans += (right - left + 1);
        set.add(new int[]{right, left});
    }

    public int count() {
        return ans;
    }

}
