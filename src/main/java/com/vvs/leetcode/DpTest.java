package com.vvs.leetcode;

import java.util.*;
import java.util.function.IntFunction;

public class DpTest {

    long[][] dp;
    int n, m;

    public long minimumTotalDistance(List<Integer> robot, int[][] factory) {
        Arrays.sort(factory, Comparator.comparingInt(o -> o[0]));
        Collections.sort(robot);
        m = robot.size();
        n = factory.length;
        this.dp = new long[n][m];
        for (int i = 0;i < n; i++) {
            Arrays.fill(dp[i], -1);
        }
        return f(0, 0, robot, factory);
    }

    public long f(int i, int j, List<Integer> robot, int[][] factory) {

        if (j >= m) {
            return 0;
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }
        if (i == n - 1) {
            if (m - j > factory[i][1]) {
                return Integer.MAX_VALUE * 1000L;
            }
            long sum = 0;
            while (j < robot.size()) {
                sum += Math.abs(robot.get(j) - factory[i][0]);
                j++;
            }
            return sum;
        }
        long res = f(i + 1, j, robot, factory);
        long sum = 0, k = 1;
        while (k <= factory[i][1] && j + k - 1 < m) {
            sum += Math.abs(robot.get((int) (j + k - 1)) - factory[i][0]);
            res = Math.min(res, sum + f(i + 1, (int) (j + k), robot, factory));
            k++;
        }
        dp[i][j] = res;
        return res;
    }

    private List<Integer>[] g;
    private Set<Long> set = new HashSet<>();
    private int k, cnt0, ans;

    public int rootCount(int[][] edges, int[][] guesses, int k) {
        this.k = k;
        g = new ArrayList[edges.length + 1];
        Arrays.setAll(g, value -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[0]].add(e[1]);
            g[e[1]].add(e[0]);
        }
        for (int[] e : guesses) {
            set.add((long) e[0] << 32 | e[1]);
        }
        dfs(0, -1);
        reboot(0, -1, cnt0);
        return ans;
    }

    private void dfs(int x, int fa) {
        for (int y : g[x]) {
            if (y != fa) {
                if (set.contains((long) x << 32 | y)) {
                    cnt0++;
                }
                dfs(y, x);
            }
        }
    }

    private void reboot(int x, int fa, int cnt) {
        if (cnt >= k) ans++;
        for (int y : g[x]) {
            if (y != fa) {
                int c = cnt;
                if (set.contains((long) x << 32 | y)) --c;
                if (set.contains((long) y << 32 | x)) ++c;
                reboot(y, x , c);
            }
        }
    }

    public static void main(String[] args) {
        List<Integer> list = Arrays.asList(1, -1);
        int[][] f = new int[][] {
                {-2,1}, {2, 1}
        };
        new DpTest().rootCount(new int[][]{
                {0,1},{1,2},{2,3},{3,4}
        }, new int[][] {
                {1,0},{3,4},{2,1},{3,2}
        }, 1);
    }
}
