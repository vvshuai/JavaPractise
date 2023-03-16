package com.vvs.leetcode;

import java.util.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 10:44 2021/11/21
 * @Modified By:
 */
public class RangeFreqQuery {

    Map<Integer, List<Integer>> map;

    public RangeFreqQuery(int[] arr) {
        map = new HashMap<>();
        for (int i = 0;i < arr.length; i++) {
            map.putIfAbsent(arr[i], new ArrayList<>());
            map.get(arr[i]).add(i);
        }
    }

    public int query(int left, int right, int value) {
        List<Integer> list = map.get(value);
        if (list == null || list.size() == 0) {
            return 0;
        }
        Collections.sort(list);
        int l = findY(list, left);
        int r = findX(list, right);
        if (l > r) {
            return 0;
        }
        return r - l + 1;
    }

    private int findX(List<Integer> list, int value) {
        int l = 0, r = list.size() - 1;
        while (l <= r) {
            int mid = l + ((r - l) >> 1);
            if (list.get(mid) > value) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return r;
    }

    private int findY(List<Integer> list, int value) {
        int l = 0, r = list.size() - 1;
        while (l <= r) {
            int mid = l + ((r - l) >> 1);
            if (list.get(mid) < value) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return l;
    }

    public static void main(String[] args) {
        RangeFreqQuery rangeFreqQuery = new RangeFreqQuery(new int[]{8,4,2,5,4,5,8,6,2,3});
        rangeFreqQuery.query(0,3,5);
        rangeFreqQuery.query(5,6,2);
        rangeFreqQuery.query(6,8,4);
        rangeFreqQuery.query(2,8,3);
        rangeFreqQuery.query(4,5,1);

    }
}
