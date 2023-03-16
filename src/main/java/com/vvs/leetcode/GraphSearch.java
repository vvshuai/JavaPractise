package com.vvs.leetcode;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class GraphSearch {

    List<int[]> ans = new ArrayList<>();
    int[][] dirs = {
            {0, 1}, {1, 0}, {0, -1}, {-1, 0}
    };

    int n, m;

    public int[][] ballGame(int num, String[] plate) {
        n = plate.length;
        m = plate[0].length();
        for (int i = 1;i < n - 1; i++) {
            if (check(i, 0, num , plate, 0)) {
                ans.add(new int[]{i, 0});
            }
            if (check(i, m - 1, num , plate, 2)) {
                ans.add(new int[]{i, m - 1});
            }
        }
        for (int j = 1;j < m - 1; j++) {
            if (check(0, j, num , plate, 1)) {
                ans.add(new int[]{0, j});
            }
            if (check(n - 1, j, num , plate, 3)) {
                ans.add(new int[]{n - 1, j});
            }
        }
        int[][] res = new int[ans.size()][2];
        for (int i = 0;i < ans.size(); i++) {
            res[i] = ans.get(i);
        }
        return res;
    }

    public boolean check(int i, int j, int num, String[] plate, int dir) {
        return dfs(i, j, num, plate, dir, 0);
    }

    public boolean dfs(int i, int j, int num, String[] plate, int dir, int cur) {
        if (cur > num) {
            return false;
        }
        if (i >= n || i < 0 || j >= m || j < 0) {
            return false;
        }
        char vv = plate[i].charAt(j);
        if (cur == 0 && (vv != '.')) {
            return false;
        }
        if (vv == 'O') {
            return true;
        }
        if (vv == '.') {
            return dfs(i + dirs[dir][0], j + dirs[dir][1], num, plate, dir, cur + 1);
        }
        if (vv == 'W') {
            dir--;
            dir = (dir + dirs.length) % dirs.length;
        } else {
            dir++;
            dir = dir % dirs.length;
        }
        return dfs(i + dirs[dir][0], j + dirs[dir][1], num, plate, dir, cur + 1);
    }

    public static void main(String[] args) {
        new GraphSearch().ballGame(4, new String[] {
                "..E.",".EOW","..W."
        });
    }
}
