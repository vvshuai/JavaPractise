package com.vvs.leetcode;

import javafx.scene.Scene;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 13:23 2022/7/24
 * @Modified By:
 */
public class MyCalendarTwo {

    TreeMap<Integer, Integer> map = new TreeMap<>();

    public MyCalendarTwo() {
    }

    public boolean book(int start, int end) {
        int ans = 0;
        int maxBook = 0;
        map.put(start, map.getOrDefault(start, 0) + 1);
        map.put(end, map.getOrDefault(end, 0) - 1);
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            int freq = entry.getValue();
            maxBook += freq;
            if (maxBook > 2) {
                map.put(start, map.getOrDefault(start, 0) - 1);
                map.put(end, map.getOrDefault(end, 0) + 1);
                return false;
            }
        }
        return true;
    }
}
