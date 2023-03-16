package com.vvs.leetcode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:59 2021/9/4
 * @Modified By:
 */
public class LockingTree {

    private int[] lock;
    private int[] parent;
    private Map<Integer, List<Integer>> map = new HashMap<>();

    public LockingTree(int[] parent) {
        lock = new int[parent.length];
        this.parent = parent;
        for (int i = 0;i < parent.length; i++) {
            map.putIfAbsent(parent[i], new ArrayList<>());
        }
        for (int i = 0;i < parent.length; i++) {
            putStatus(parent[i], i);
        }
    }

    private void putStatus(int num, int i) {
        while (num != -1) {
            map.get(num).add(i);
            num = parent[num];
        }
    }

    public boolean lock(int num, int user) {
        if (lock[num] == 0) {
            lock[num] = user;
            return true;
        }
        return false;
    }

    public boolean unlock(int num, int user) {
        if (lock[num] == user) {
            lock[num] = 0;
            return true;
        }
        return false;
    }

    public boolean upgrade(int num, int user) {
        boolean cur = find(parent[num]);
        if (!cur) {
            return false;
        }
        if (lock[num] != 0) {
            return false;
        }
        List<Integer> curList = map.get(num);
        boolean flag = false;
        if (curList == null) {
            return false;
        }
        for (int x : curList) {
            if (lock[x] != 0) {
                flag = true;
                lock[x] = 0;
            }
        }
        if (!flag) {
            return false;
        }
        lock[num] = user;
        return true;
    }

    public boolean find(int num) {
        while (num != -1) {
            if (lock[num] != 0) {
                return false;
            }
            num = parent[num];
        }
        return true;
    }

    public static void main(String[] args) {
        LockingTree lockingTree = new LockingTree(new int[]{-1, 0, 3, 1, 0});
        System.out.println(lockingTree.upgrade(4,5));
    }
}
