package com.vvs.leetcode;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:41 2022/7/23
 * @Modified By:
 */
public class NumberContainers {

    Map<Integer, TreeSet<Integer>> map;
    Map<Integer, Integer> vvmap;

    public NumberContainers() {
        map = new HashMap<>();
        vvmap = new HashMap<>();
    }

    public void change(int index, int number) {
        map.putIfAbsent(number, new TreeSet<>());
        if (vvmap.get(index) != null) {
            int cur = vvmap.get(index);
            map.get(cur).remove(index);
        }
        map.get(number).add(index);
        vvmap.put(index, number);
    }

    public int find(int number) {
        if (map.get(number) == null || map.get(number).size() == 0) {
            return -1;
        }
        return map.get(number).ceiling(Integer.MIN_VALUE);
    }

    public static void main(String[] args) {
        NumberContainers nc = new NumberContainers();
        nc.change(1, 10); // 容器中下标为 2 处填入数字 10 。
        nc.change(1, 10); // 容器中下标为 1 处填入数字 10 。
        int i = nc.find(10); // 数字 10 所在的下标为 1 ，2 ，3 和 5 。因为最小下标为 1 ，所以返回 1 。
        System.out.println(i);
    }
}
