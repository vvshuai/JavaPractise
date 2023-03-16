package com.vvs.leetcode;

import java.util.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:37 2022/7/5
 * @Modified By:
 */
public class MyCalendar {

    private TreeMap<Integer, Integer> map;

    public MyCalendar() {
        map = new TreeMap<>();
        map.put(-1, -1);
        map.put(Integer.MAX_VALUE, Integer.MAX_VALUE);
    }

    public boolean book(int start, int end) {
        int key1 = map.ceilingKey(start);
        int key2 = map.floorKey(start);
        if (end < key1 && map.get(key2) < start) {
            map.put(start, end - 1);
            return true;
        }
        return false;
    }
}
