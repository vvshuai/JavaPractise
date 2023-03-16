package com.vvs.leetcode;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import java.util.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 16:25 2022/7/23
 * @Modified By:
 */
public class LargeSimulation {

    static int[][] dirs = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};

    public int containVirus(int[][] isInfected) {
        int m = isInfected.length;
        int n = isInfected[0].length;
        int ans = 0;
        while (true) {
            List<Set<Integer>> neighbors = new ArrayList<>();
            List<Integer> firewalls = new ArrayList<>();
            for (int i = 0;i < m; i++) {
                for (int j = 0;j < n; j++) {
                    // 每个联通块分别计数
                    if (isInfected[i][j] == 1) {
                        Queue<int[]> que = new ArrayDeque<>();
                        que.offer(new int[]{i, j});
                        Set<Integer> set = new HashSet<>();
                        int firewall = 0;
                        int idx = neighbors.size() + 1;
                        isInfected[i][j] = -idx;
                        while (!que.isEmpty()) {
                            int[] cur = que.poll();
                            for (int[] d : dirs) {
                                int nx = cur[0] + d[0];
                                int ny = cur[1] + d[1];
                                if (nx < 0 || nx >= m || ny < 0 || ny >= n) {
                                    continue;
                                }
                                if (isInfected[nx][ny] == 1) {
                                    que.offer(new int[]{nx, ny});
                                    isInfected[nx][ny] = -idx;
                                } else if (isInfected[nx][ny] == 0) {
                                    firewall++;
                                    set.add(getHash(nx, ny));
                                }
                            }
                        }
                        neighbors.add(set);
                        firewalls.add(firewall);
                    }
                }
            }
            if (neighbors.isEmpty()) {
                break;
            }
            int idx = 0;
            for (int i = 1;i < neighbors.size(); i++) {
                if (neighbors.get(i).size() > neighbors.get(idx).size()) {
                    idx = i;
                }
            }
            ans += firewalls.get(idx);
            for (int i = 0;i < m; i++) {
                for (int j = 0;j < n; j++) {
                    if (isInfected[i][j] < 0) {
                        if (isInfected[i][j] == (-idx - 1)) {
                            isInfected[i][j] = 8;
                        } else {
                            isInfected[i][j] = 1;
                        }
                    }
                }
            }
            for (int i = 0;i < neighbors.size(); i++) {
                if (i != idx) {
                    for (int hash : neighbors.get(i)) {
                        int x = hash >> 16;
                        int y = hash & ((1 << 16) - 1);
                        isInfected[x][y] = 1;
                    }
                }
            }
            if (neighbors.size() == 1) {
                break;
            }
        }
        return ans;
    }

    public int getHash(int x, int y) {
        return (x << 16) ^ y;
    }

}
