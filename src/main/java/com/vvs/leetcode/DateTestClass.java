package com.vvs.leetcode;

import java.text.SimpleDateFormat;
import java.time.Instant;
import java.util.*;
/**
 * @Author: vvshuai
 * @Description: 日期类测试
 * @Date: Created in 16:17 2021/11/14
 * @Modified By:
 */
public class DateTestClass {

    public static void main(String[] args) {
        findLHS(new int[]{1,2,3,3,1,-14,13,4});
    }

    public static int findLHS(int[] nums) {
        Map<Integer, Integer> map = new TreeMap<>();
        int ans = Integer.MIN_VALUE;
        int last = 0;
        for (int i = 0;i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        boolean flag = false;
        for (int x : map.keySet()) {
            if (!flag) {
                last = x;
                flag = true;
            } else {
                if (x - last == 1) {
                    ans = Math.max(ans, map.get(last) + map.get(x));
                    last = x;
                }
            }
        }
        return ans;
    }
}
