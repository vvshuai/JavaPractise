package com.vvs.leetcode;

import com.sun.jmx.remote.internal.ArrayQueue;
import com.vvs.algorithm.BITree;

import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.util.stream.IntStream;

public class Main {

    /**
     * @Description: 1743. 从相邻元素对还原数组
     * @return: int[]
     */
    public static int[] restoreArray(int[][] adjacentPairs) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] arr : adjacentPairs) {
            map.putIfAbsent(arr[0], new ArrayList<>());
            map.putIfAbsent(arr[1], new ArrayList<>());
            map.get(arr[0]).add(arr[1]);
            map.get(arr[1]).add(arr[0]);
        }

        int n = adjacentPairs.length + 1;
        int[] ans = new int[n];
        for (int x : map.keySet()) {
            if (map.get(x).size() == 1) {
                ans[0] = x;
                break;
            }
        }
        ans[1] = map.get(ans[0]).get(0);
        for (int i = 2; i < n; i++) {
            List<Integer> list = map.get(ans[i - 1]);
            ans[i] = ans[i - 2] == list.get(0) ? list.get(0) : list.get(1);
        }
        return ans;
    }

    public int maxArea(int[] a) {
        int ans = 0;
        int l = 0, r = a.length - 1;
        while (l < r) {
            int lc = a[l];
            int rc = a[r];
            ans = Math.max(ans, Math.min(lc, rc) * (r - l));
            if (lc < rc) {
                l++;
            } else {
                r++;
            }
        }
        return ans;
    }

    public static long numberOfWeeks(int[] milestones) {
        long ans = 0;
        int n = milestones.length;
        Map<Integer, Integer> map = new TreeMap<>();
        for (int i = 0; i < n; i++) {
            map.put(milestones[i], map.getOrDefault(milestones[i], 0) + 1);
        }
        int cur = 0;
        int i = 1;
        for (int x : map.keySet()) {
            if (i == map.size()) {
                if (map.get(x) == 1) {
                    if (x - cur > 2) {
                        ans += 2;
                    } else {
                        ans += (x - cur);
                    }
                } else {
                    ans += (x - cur) * map.get(x);

                }
                break;
            }
            ans += (x - cur) * n;
            n -= map.get(x);
            cur = x;
            i++;
        }
        return ans;
    }

    public List<List<Integer>> threeSum(int[] nums, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;
        for (int zero = 0; zero < n; zero++) {
            if (zero > 0 && nums[zero] == nums[zero - 1]) {
                continue;
            }
            for (int first = zero + 1; first < n; first++) {
                if (first > zero + 1 && nums[first] == nums[first - 1]) {
                    continue;
                }
                int vv = target - nums[zero] - nums[first];
                int third = n - 1;
                for (int second = first + 1; second < n; second++) {
                    if (second > first + 1 && nums[second] == nums[second - 1]) {
                        continue;
                    }
                    while (second < third && nums[second] + nums[third] > vv) {
                        third--;
                    }
                    if (second == third) {
                        break;
                    }
                    if (nums[second] + nums[third] == target) {
                        List<Integer> list = new ArrayList<>();
                        list.add(nums[zero]);
                        list.add(nums[first]);
                        list.add(nums[second]);
                        list.add(nums[third]);
                        ans.add(list);
                    }
                }
            }
        }

        return ans;
    }

    public static void nextPermutation(int[] nums) {
        int n = nums.length;
        for (int i = n - 1; i > 0; i--) {
            int curA = nums[i];
            int curB = nums[i - 1];
            if (curB < curA) {
                for (int j = n - 1; j >= i; j--) {
                    if (nums[j] > curB) {
                        int t = nums[j];
                        nums[j] = curB;
                        nums[i - 1] = t;
                        Arrays.sort(nums, i, n);
                        return;
                    }
                }
            }
        }
        Arrays.sort(nums);
    }

    public static int minStoneSum(int[] piles, int k) {
        Queue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });
        for (int i = 0; i < piles.length; i++) {
            queue.add(piles[i]);
        }
        while (k-- > 0) {
            double x = queue.poll();
            queue.add((int) Math.ceil(x / 2));
        }
        int ans = 0;
        while (!queue.isEmpty()) {
            ans += queue.poll();
        }
        return ans;
    }

    public int[] longestObstacleCourseAtEachPosition(int[] obstacles) {
        int n = obstacles.length;
        List<Integer> d = new ArrayList<>();
        List<Integer> ans = new ArrayList<>();
        for (int ob : obstacles) {
            if (d.isEmpty() || ob >= d.get(d.size())) {
                d.add(ob);
                ans.add(d.size());
            } else {
                int loc = upper_bound(d, ob);
                ans.add(loc + 1);
                d.set(loc, ob);
            }
        }
        int[] cur = new int[n];
        for (int i = 0; i < n; i++) {
            cur[i] = ans.get(i);
        }
        return cur;
    }

    public int upper_bound(List<Integer> list, int target) {
        int l = 0;
        int r = list.size();
        while (l <= r) {
            int mid = (l + r) >> 1;
            if (list.get(mid) > target) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    public int[] rearrangeArray(int[] nums) {
        for (int i = 1; i < nums.length - 1; i++) {
            int front = nums[i - 1];
            int back = nums[i + 1];
            double curNum = (front + back) / 2.0;
            if (curNum == (double) nums[i]) {
                int t = nums[i];
                nums[i] = nums[i + 1];
                nums[i + 1] = t;
            }
        }
        return nums;
    }

    /**
     * @Description: 并查集dfs
     * @return: void
     */
    private static int[] parent = new int[100100];
    private static int[] rank = new int[100100];

    public static void init(int n) {
        for (int i = 0; i <= n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }
    }

    public static int find(int x) {
        if (parent[x] == x) {
            return x;
        }
        return parent[x] = find(parent[x]);
    }

    public static void unite(int u, int v) {
        int a = find(u);
        int b = find(v);
        if (a == b) {
            return;
        }
        if (rank[a] < rank[b]) {
            parent[a] = b;
        } else {
            parent[b] = a;
            if (rank[a] == rank[b]) {
                rank[a]++;
            }
        }
    }

    public static boolean same(int a, int b) {
        return find(a) == find(b);
    }


    public static int latestDayToCross(int row, int col, int[][] cells) {
        int n = row * col;
        init(n + 2);
        int ans = 0;
        boolean[][] valid = new boolean[row][col];
        for (int i = n - 1; i >= 0; --i) {
            int x = cells[i][0] - 1, y = cells[i][1] - 1;
            valid[x][y] = true;
            int id = x * col + y;
            if (x - 1 >= 0 && valid[x - 1][y]) {
                unite(id, id - col);
            }
            if (x + 1 < row && valid[x + 1][y]) {
                unite(id, id + col);
            }
            if (y - 1 >= 0 && valid[x][y - 1]) {
                unite(id, id - 1);
            }
            if (y + 1 < col && valid[x][y + 1]) {
                unite(id, id + 1);
            }
            if (x == 0) {
                unite(id, n);
            }
            if (x == row - 1) {
                unite(id, n + 1);
            }
            if (same(n, n + 1)) {
                ans = i;
                break;
            }
        }
        return ans;
    }

    public String reverseVowels(String s) {
        int l = 0;
        int r = s.length();
        char[] arr = s.toCharArray();
        while (l < r) {
            if (judge(arr[l]) && judge(arr[r])) {
                l++;
                r--;
            } else if (judge(arr[l])) {
                r--;
            } else if (judge(arr[r])) {
                l++;
            }
        }
        return String.valueOf(arr);
    }

    public boolean judge(char x) {
        if (x == 'a' || x == 'e' || x == 'i' || x == 'o' || x == 'u') {
            return true;
        }
        if (x == 'A' || x == 'E' || x == 'I' || x == 'O' || x == 'U') {
            return true;
        }
        return false;
    }

    public static int compress(char[] chars) {
        int l = 0, r = 0;
        int ans = 0;
        int index = 0;
        while (l < chars.length) {
            while (chars[l] == chars[r]) {
                r++;
            }
            ans++;
            String s = String.valueOf(r - l);
            chars[index++] = chars[l];
            if (r - l != 1) {
                for (int i = 0; i < s.length(); i++) {
                    chars[index++] = s.charAt(i);
                    ans++;
                }
            }
            l = r;
        }
        return ans;
    }

    public int minTimeToType(String word) {
        int ans = 0;
        char last = 'a';
        for (int i = 0; i < word.length(); i++) {
            ans += deal(last, word.charAt(i));
            ans++;
            last = word.charAt(i);
        }
        return ans;
    }

    public int deal(char x, char y) {
        int sum1 = 0;
        int sum2 = 0;
        if (x == y) {
            return 0;
        }
        if (x > y) {
            char t = x;
            x = y;
            y = t;
        }
        sum1 = y - x;
        sum2 = ('z' - y) + (x - 'a') + 1;
        return Math.min(sum1, sum2);
    }

    private long inf = 0x3f3f3f3f3fL;
    private long[][] map = new long[251][251];
    private long[] dis = new long[500];
    private long[] vis = new long[500];
    private long[] tot = new long[500];
    private int mod = 1000000007;


    public int countPaths(int n, int[][] roads) {
        for (int i = 0; i <= n; i++) {
            Arrays.fill(map[i], inf);
        }
        for (int i = 0; i < roads.length; i++) {
            int a = roads[i][0] + 1;
            int b = roads[i][1] + 1;
            int c = roads[i][2];
            map[a][b] = c;
            map[b][a] = c;
        }
        for (int i = 1; i <= n; i++) {
            map[i][i] = 0;
        }
        dijkstra(n);
        return (int) tot[n];
    }

    public void dijkstra(int n) {
        Arrays.fill(vis, 0);
        for (int i = 0; i <= n; i++) {
            tot[i] = 0;
        }
        tot[1] = 1;
        Arrays.fill(dis, inf);
        dis[1] = 0;
        for (int i = 1; i <= n; i++) {
            long minn = inf;
            int pos = -1;
            for (int j = 1; j <= n; j++) {
                if (vis[j] == 0 && dis[j] < minn) {
                    minn = dis[j];
                    pos = j;
                }
            }
            vis[pos] = 1;
            for (int j = 1; j <= n; j++) {
                if (j == pos) {
                    continue;
                }
                if (dis[pos] + map[pos][j] == dis[j]) {
                    tot[j] += tot[pos];
                    tot[j] %= mod;
                } else if (dis[pos] + map[pos][j] < dis[j]) {
                    tot[j] = tot[pos];
                    tot[j] %= mod;
                    dis[j] = dis[pos] + map[pos][j];
                }
            }
        }
    }

    public long maxMatrixSum(int[][] matrix) {
        int ans = 0;
        int min = Integer.MAX_VALUE;
        int cur = 0;
        Queue<Integer> que = new PriorityQueue<Integer>();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                int x = Math.abs(matrix[i][j]);
                min = Math.min(x, min);
                if (matrix[i][j] <= 0) {
                    cur++;
                }
                ans += matrix[i][j];
            }
        }
        if (cur % 2 == 1) {
            ans += (min * (-1));
        }
        return ans;
    }

    public boolean escapeGhosts(int[][] ghosts, int[] target) {
        int[] source = new int[]{0, 0};
        int dis = distance(source, target);
        for (int i = 0; i < ghosts.length; i++) {
            int newDis = distance(ghosts[i], target);
            if (newDis < dis) {
                return false;
            }
        }
        return true;
    }

    public int distance(int[] arr1, int[] arr2) {
        return Math.abs(arr1[0] - arr2[0]) + Math.abs(arr1[1] - arr2[1]);
    }

    public String findDifferentBinaryString(String[] nums) {
        Map<Integer, String> map = new TreeMap<>();
        int n = nums.length;
        for (String s : nums) {
            char[] chars = s.toCharArray();
            int ans = 0;
            int cur = 1;
            for (int i = chars.length - 1; i >= 0; i++) {
                ans += (chars[i] - '0') * cur;
                cur *= 2;
            }
            map.put(ans, s);
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 65536; i++) {
            if (!map.containsKey(i)) {
                int x = i;
                while (x != 0) {
                    sb.append(x % 2);
                    x /= 2;
                }
                for (int j = sb.length(); j < n; j++) {
                    sb.append('0');
                }
                break;
            }
        }
        sb.reverse();
        return sb.toString();
    }

    public int minimizeTheDifference(int[][] mat, int target) {
        for (int i = 0; i < mat.length; i++) {
            Arrays.sort(mat[i]);
        }
        int[][] inArr = new int[mat.length][mat[0].length];
        int[] par = new int[mat.length];
        for (int i = 0; i < mat.length; i++) {
            par[i] = 1;
        }
        for (int i = 0; i < mat.length; i++) {
            for (int j = 1; j < mat[0].length; j++) {
                inArr[i][j] = mat[i][j] - mat[i][0];
            }
        }
        int cur = 0;
        int ans = 0x3f3f3f3f;
        for (int j = 0; j < mat.length; j++) {
            cur += mat[j][0];
        }
        ans = Math.min(ans, Math.abs(cur - target));
        while (true) {
            int min = Integer.MAX_VALUE;
            int d = -1;
            for (int j = 0; j < mat.length; j++) {

                if (par[j] < mat[0].length && inArr[j][par[j]] < min) {
                    min = inArr[j][par[j]];
                    d = j;
                }
            }
            if (d == -1) {
                break;
            } else {
                cur -= mat[d][par[d] - 1];
                cur += mat[d][par[d]];
                par[d]++;
                if (Math.abs(cur - target) <= ans) {
                    ans = Math.abs(cur - target);
                } else {
                    return ans;
                }
            }
        }
        return ans;
    }

    Map<Integer, List<Integer>> ma = new HashMap<>();
    private List<List<Integer>> ans1 = new ArrayList<>();
    private int n;

    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        n = graph.length;
        List<Integer> cur = new ArrayList<>();
        for (int i = 0;i < graph.length; i++) {
            for (int j = 0;j < graph[i].length; j++) {
                ma.putIfAbsent(i, new ArrayList<>());
                ma.get(i).add(graph[i][j]);
            }
        }
        dfs(0, cur);
        return ans1;
    }

    public void dfs(int index, List<Integer> cur) {
        cur.add(index);
        if (index == n - 1) {
            ans1.add(new ArrayList<>(cur));
            return ;
        }
        List<Integer> list = ma.get(index);
        for (int i = 0;i < list.size(); i++) {
            dfs(list.get(i), cur);
            cur.remove(cur.size()-1);
        }
    }

    public int minSessions(int[] tasks, int sessionTime) {
        int[] dp = new int[20];
        int inf = 20;
        int n = tasks.length;
        int m = (1 << n);
        Arrays.fill(dp, inf);
        for (int i = 1;i < m; i++) {
            int cur = 0;
            int idx = 0;
            int x = i;
            while (x > 0) {
                int b = x & 1;
                if (b == 1) {
                    cur += tasks[idx];
                }
                x >>= 1;
            }
            if (cur <= sessionTime) {
                dp[i] = 1;
            }
        }
        for (int i = 1;i < m; i++) {
            for (int j = i;j > 0; j = ((j - 1) & i)) {
                dp[i] = Math.min(dp[i], dp[j] + dp[i ^ j]);
            }
        }
        return dp[m-1];
    }

    public int compareVersion(String version1, String version2) {
        String[] v1s = version1.split("\\.");
        String[] v2s = version2.split("\\.");
        for (int i = 0;i < v1s.length || i < v2s.length; i++) {
            int x = 0, y = 0;
            if (i < v1s.length) {
                x = Integer.parseInt(v1s[i]);
            }
            if (i < v2s.length) {
                y = Integer.parseInt(v2s[i]);
            }
            if (x > y) {
                return 1;
            }
            if (x < y) {
                return -1;
            }
        }
        return 0;
    }

    public int findMiddleIndex(int[] nums) {
        int n = nums.length;
        if (n == 1) {
            return 0;
        }
        for (int i = 0;i < n; i++) {
            int cur1 = 0;
            int cur2 = 0;
            for (int j = 0;j < n;j ++) {
                if (j < i) {
                    cur1 += nums[j];
                } else if (j > i) {
                    cur2 += nums[j];
                }
            }
            if (cur1 == cur2) {
                return i;
            }
        }
        return -1;
    }

    public int[][] findFarmland(int[][] land) {
        int m = land.length;
        int n = land[0].length;
        List<int[]> curList = new ArrayList<>();
        Queue<Pair> que = new LinkedList<>();
        int[][] vis = new int[m][n];
        for (int i = 0;i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (land[i][j] == 1 && vis[i][j] == 0) {
                    int[] curArr = new int[4];
                    curArr[0] = i;
                    curArr[1] = j;
                    que.add(new Pair(i, j));
                    while (!que.isEmpty()) {
                        Pair cur = que.poll();
                        if (vis[cur.x][cur.y] == 1) {
                            continue;
                        }
                        curArr[2] = cur.x;
                        curArr[3] = cur.y;
                        vis[cur.x][cur.y] = 1;
                        if (cur.x + 1 < n && land[cur.x+1][cur.y] == 1) {
                            que.add(new Pair(cur.x + 1, cur.y));
                        }
                        if (cur.x - 1 >= 0 && land[cur.x-1][cur.y] == 1) {
                            que.add(new Pair(cur.x - 1, cur.y));
                        }
                        if (cur.y + 1 < m && land[cur.x][cur.y+1] == 1) {
                            que.add(new Pair(cur.x, cur.y + 1));
                        }
                        if (cur.y - 1 >= 0 && land[cur.x][cur.y-1] == 1) {
                            que.add(new Pair(cur.x, cur.y - 1));
                        }
                    }
                    curList.add(curArr);
                }
            }
        }
        int[][] ans = new int[curList.size()][4];
        int v = 0;
        for (int[] x : curList) {
            ans[v++] = x;
        }
        return ans;
    }

    class Pair {
        int x, y;

        public Pair(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    public int firstDayBeenInAllRooms(int[] nextVisit) {
        int n = nextVisit.length;
        int[] dp = new int[n+1];
        for (int i = 1;i < n; i++) {
            dp[i] = (2 * dp[i-1] - dp[nextVisit[i-1] + 2 + mod])%mod;
        }
        return dp[n-2];
    }

    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
        int n = profits.length;
        int[][] arr = new int[n][2];
        for (int i = 0;i < n; i++) {
            arr[i][0] = profits[i];
            arr[i][1] = capital[i];
        }
        int cur = 0;
        Arrays.sort(arr, Comparator.comparingInt(a -> a[0]));
        Queue<Integer> que = new PriorityQueue<>((x, y) -> (y - x));
        for (int i = 0;i < k; i++) {
            while (cur < n && arr[cur][0] <= w) {
                que.add(arr[cur][1]);
                cur++;
            }
            if (!que.isEmpty()) {
                w += que.poll();
            } else {
                break;
            }
        }
        return w;
    }

    public int minimumSwitchingTimes(int[][] source, int[][] target) {
        int[] arr1 = new int[10010];
        for (int i = 0;i < source.length; i++) {
            for (int j = 0;j < source[0].length; j++) {
                arr1[source[i][j]]++;
            }
        }
        for (int i = 0;i < target.length; i++) {
            for (int j = 0;j < target[0].length; j++) {
                arr1[target[i][j]]--;
            }
        }
        int ans = 0;
        for (int i = 1;i <= 10000; i++) {
            if (arr1[i] != 0) {
                ans++;
            }
        }
        return ans >> 1;
    }

    public int maxmiumScore(int[] cards, int cnt) {
        Arrays.sort(cards);
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        for (int i = 0;i < cards.length; i++) {
            if (cards[i] % 2 == 0) {
                list1.add(cards[i]);
            } else {
                list2.add(cards[i]);
            }
        }
        int ans = 0;
        int l = list1.size();
        int r = list2.size();
        if (cnt % 2 != 0) {
            if (l != 0) {
                ans += list1.get(l - 1);
                l--;
                cnt--;
            }
        }
        if (cnt % 2 == 0 && cnt != 0) {
            while (l - 2 >= 0 || r - 2 >= 0){
                if (cnt == 0) {
                    break;
                }
                int vv1 = l - 2 >= 0 ? list1.get(l - 1) + list1.get(l-2) : 0;
                int vv2 = r - 2 >= 0 ? list2.get(r - 1) + list2.get(r-2) : 0;
                if (vv1 > vv2) {
                    ans += vv1;
                    l -= 2;
                } else {
                    ans += vv2;
                    r -= 2;
                }
                cnt -= 2;
            }
        }
        if (cnt == 0) {
            return ans;
        }
        return 0;
    }

    public boolean checkValidString(String s) {
        char[] chars = s.toCharArray();
        int cur1 = 0;
        int cur2 = 0;
        for (int i = 0;i < chars.length; i++) {
            char x = chars[i];
            if (x == '(') {
                cur1++;
                cur2++;
            } else if (x == ')') {
                cur1 = Math.max(cur1 - 1, 0);
                cur2--;
                if (cur2 < 0) {
                    return false;
                }
            } else {
                cur1 = Math.min(cur1 - 1, 0);
                cur2++;
            }
        }
        return cur1 == 0;
    }

    public String reversePrefix(String word, char ch) {
        int n = word.length();
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        int flag = 0;
        for (int i = 0;i < n; i++) {
            sb1.append(i);
            if (word.charAt(i) == ch && flag == 0) {
                sb1.reverse();
                flag = 1;
            }
        }
        return sb1.toString();
    }

    public long interchangeableRectangles(int[][] rectangles) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0;i < n; i++) {
            int x = gcd(rectangles[i][0], rectangles[i][1]);
            StringBuilder sb = new StringBuilder();
            sb.append(rectangles[i][0]/x).append(",").append(rectangles[i][1]/x);
            String key = sb.toString();
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        long ans = 0;
        for (String key : map.keySet()) {
            int value = map.get(key);
            int n = value - 1;
            ans += n + (n*(n-1))/2;
        }
        return ans;
    }

    public int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    public int maxProduct(String s) {
        int n = s.length();
        int v = (1 << n);
        List<Pair> list = new ArrayList<>();
        for (int i = 0;i < v; i++) {
            int cur = i;
            StringBuilder sb = new StringBuilder();
            for (int j = 0;j < n; j++) {
                if ((cur & 1) == 1) {
                    sb.append(s.charAt(j));
                }
                cur >>= 1;
            }
            if (check(sb) && sb.length() != n) {
                list.add(new Pair(i, sb.length()));
            }
        }
        int ans = 0;
        for (int i = 0;i < list.size(); i++) {
            for (int j = i+1;j < list.size(); j++) {
                int x = list.get(i).x;
                int y = list.get(j).x;
                if ((x & y)== 0) {
                    ans = Math.max(ans, list.get(i).y * list.get(j).y);
                }
            }
        }
        return ans;
    }

    public boolean isOneBitCharacter(int[] bits) {
        int i=0;
        while(i<bits.length-1){i+=bits[i]==1?2:1;}
        return i==bits.length-1;
    }

    private boolean check(StringBuilder sb) {
        if (sb.length() == 0) {
            return false;
        }
        for (int i = 0;i < sb.length(); i++) {
            if (sb.charAt(i) != sb.charAt(sb.length() - 1 - i)) {
                return false;
            }
        }
        return true;
    }

    public int numberOfBoomerangs(int[][] points) {
        int ans = 0;
        for (int[] p : points) {
            Map<Integer, Integer> map = new HashMap<>();
            for (int[] q : points) {
                int dis = ((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]));
                map.put(dis, map.getOrDefault(dis, 0) + 1);
            }
            for (Integer k : map.keySet()) {
                int m = map.get(k);
                ans += (m-1) * m;
            }
        }
        return ans;
    }

    public String findLongestWord(String s, List<String> dictionary) {
        dictionary.sort(new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                if (o1.length() != o2.length()) {
                    return o2.length() - o1.length();
                }
                return o1.compareTo(o2);
            }
        });
        for (String v : dictionary) {
            int i = 0, j = 0;
            while (i < s.length()) {
                if (s.charAt(i) == v.charAt(j)) {
                    i++;
                    j++;
                } else {
                    i++;
                }
            }
            if (j == v.length()) {
                return v;
            }
        }
        return "";
    }

    public int findPeakElement(int[] nums) {
        int n = nums.length;
        int l = 0, r = n-1, ans = -1;
        while (l < r) {
            // 随机选择一个位置
            int mid = (l + r) >> 1;
            if (nums[mid] > nums[mid+1]) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    public int lengthOfLastWord(String s) {
        String[] strings = s.split(" ");
        int max = Integer.MIN_VALUE;
        for (int i = 0;i < strings.length; i++) {
            if (strings[i].length() == 0) {
                continue;
            }
            max = Math.max(max, strings[i].length());
        }
        return max;
    }

    public int minDistance(String word1, String word2) {
        int[][] dp = new int[505][505];
        int n = word1.length();
        int m = word2.length();
        for (int i = 1;i <= word1.length(); i++) {
            for (int j = 1;j <= word2.length(); j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        return dp[n][m];
    }

    public int arrangeCoins(int n) {
        int cur = 1;
        int ans = 0;
        while (n - cur <= 0) {
            ans ++;
            n -= cur;
            cur++;
        }
        return ans;
    }

    public int distributeCandies(int[] candyType) {
        Set<Integer> set = new HashSet<>();
        for (int x : candyType) {
            set.add(x);
        }
        return Math.max(set.size(), candyType.length >> 1);
    }

    public boolean detectCapitalUse(String word) {
        char[] words = word.toCharArray();
        int flag1 = 0;
        int flag2 = 0;
        for (int i = 0;i < words.length; i++) {
            if (words[i] >= 'A' && words[i] <= 'Z') {
                flag1++;
            } else {
                flag2++;
            }
        }
        return (flag1 == words.length || flag2 == words.length) || ((words[0] >= 'A' && words[0] <= 'Z') && flag2 == words.length - 1);
    }

    public int maxDistance(int[] colors) {
        int ans = 0;
        for (int i = 0;i < colors.length; i++) {
            for (int j = i + 1;j < colors.length; j++) {
                if (colors[j] != colors[i]) {
                    ans = Math.max(ans, j - i);
                }
            }
        }
        return ans;
    }

    public int wateringPlants(int[] plants, int capacity) {
        int ans = 0;
        int cur = capacity;
        for (int i = 0;i < plants.length; i++) {
            if (capacity < plants[i]) {
                ans += i + (i + 1);
                capacity = cur;
                capacity -= plants[i];
            } else {
                capacity -= plants[i];
                ans++;
            }
        }
        return ans;
    }

    public boolean buddyStrings(String s, String goal) {
        System.out.println(s.length());
        System.out.println(goal.length());
        char[] chars = s.toCharArray();
        char[] chars1 = goal.toCharArray();
        Map<Character, Integer> map1 = new HashMap<>();
        Map<Character, Integer> map2 = new HashMap<>();
        int s1 = 0, e1 = 0;
        boolean flag = false;
        int cur = 0;
        for (int i = 0;i < chars.length; i++) {
            if (chars[i] != chars1[i] && cur == 0) {
                s1 = i;
                cur++;
            } else if (chars[i] != chars1[i] && cur == 1) {
                cur++;
                e1 = i;
                flag = true;
            } else if (chars[i] != chars1[i]) {
                cur++;
            }
            if (cur == 3) {
                return false;
            }
            map1.put(chars[i], map1.getOrDefault(chars[i], 0) + 1);
            map2.put(chars1[i], map2.getOrDefault(chars1[i], 0) + 1);
        }
        System.out.println(1);
        if (flag) {
            char t = chars[s1];
            chars[s1] = chars[e1];
            chars[e1] = t;
            return new String(chars).equals(new String(chars1));
        } else {
            for (char x : map1.keySet()) {
                if (map1.get(x).equals(map2.get(x)) && map1.get(x) != 1) {
                    return false;
                }
                if (map2.get(x) == null || !map1.get(x).equals(map2.get(x))) {
                    return false;
                }
            }
            return true;
        }
    }

    public int minimumBuckets(String street) {
        if (street.length() == 1 && street.charAt(0) == 'H') {
            return -1;
        }
        if (street.length() == 2 && street.charAt(0) == 'H' && street.charAt(1) == 'H') {
            return -1;
        }
        if (street.length() == 2 && street.charAt(0) == 'H' && street.charAt(1) == '.') {
            return 1;
        }
        if (street.length() == 2 && street.charAt(0) == '.' && street.charAt(1) == 'H') {
            return 1;
        }
        boolean haveH = false;
        int[] vis = new int[street.length()];
        int cur = 0;
        int ans = 0;
        for (int i = 0;i < street.length(); i++) {
            boolean haveB = false;
            if (i == 0 || i == street.length() - 1) {
                cur = 1;
            }
            if (street.charAt(i) == 'H') {
                if (i - 1 >= 0 && vis[i - 1] == 1) {
                    continue;
                }
                if (i + 1 < street.length()) {
                    if (street.charAt(i + 1) == 'H') {
                        cur++;
                    } else {
                        if (vis[i + 1] == 0) {
                            ans++;
                            vis[i + 1] = 1;
                            haveB = true;
                        }
                    }
                }
                if (i - 1 >= 0) {
                    if (street.charAt(i - 1) == 'H') {
                        cur++;
                    } else if (!haveB){
                        if (vis[i - 1] == 0) {
                            ans++;
                            vis[i - 1] = 1;
                        }
                    }
                }
            }
            if (cur == 2) {
                return -1;
            }
            cur = 0;
        }
        return ans;
    }

    public int minCost(int[] startPos, int[] homePos, int[] rowCosts, int[] colCosts) {
        if (startPos[0] == homePos[0] && startPos[1] == homePos[1]) {
            return 0;
        }
        int s, e;
        if (startPos[0] < homePos[0]) {
            s = startPos[0] + 1;
            e = homePos[0];
        } else {
            s = homePos[0];
            e = startPos[0] - 1;
        }
        int ans = 0;
        for (int i = s;i <= e; i++) {
            ans += rowCosts[i];
        }
        if (startPos[1] < homePos[1]) {
            s = startPos[1] + 1;
            e = homePos[1];
        } else {
            s = homePos[1];
            e = startPos[1] - 1;
        }
        for (int i = s;i <= e; i++) {
            ans += colCosts[i];
        }
        return ans;
    }

    private int ans = 0;

    public int countPyramids(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int ans = 0;
        for (int i = 0;i < m; i++) {
            for (int j = 0;j < n; j++) {
                if (grid[i][j] == 1) {
                    dfs1(grid, i+1, j, 3);
                    dfs2(grid, i-1, j, 3);
                }
            }
        }
        return ans;
    }

    public void dfs1(int[][] grid, int x, int y, int cur) {
        for (int i = x;i < grid.length; i++) {
            for (int j = 1;j <= ((cur - 1) >> 1) ; j++) {
                if (y - j < 0 || y + j >= grid[0].length) {
                    return ;
                }
                if (grid[i][y - j] == 1 && grid[i][y + j] == 1) {
                    ans++;
                    dfs1(grid, x+1, y, cur + 2);
                }
            }
        }
    }

    public void dfs2(int[][] grid, int x, int y, int cur) {
        for (int i = x;i >= 0; i--) {
            for (int j = 1;j <= ((cur - 1) >> 1) ; j++) {
                if (y - j < 0 || y + j >= grid[0].length) {
                    return ;
                }
                if (grid[i][y - j] == 1 && grid[i][y + j] == 1) {
                    ans++;
                    dfs2(grid, x-1, y, cur + 2);
                }
            }
        }
    }

    public int[] kthSmallestPrimeFraction(int[] arr, int k) {
        int n = arr.length;
        double l = 0, r = 1;
        while (true) {
            double mid = (l + r) / 2;
            int i = -1, count = 0;
            int x = 0, y = 1;
            for (int j = 1;j < n; j++) {
                while (arr[i + 1] * 1.0 / arr[j] < mid) {
                    i++;
                    x = arr[i];
                    y = arr[j];
                }
                count += i + 1;
            }
            if (count == k) {
                return new int[]{x, y};
            }
            if (count < k) {
                 l = mid;
            } else {
                r = mid;
            }
        }
    }

    public int findNthDigit(int n) {
        long num = 9;
        int d = 1;
        while (n > d * num) {
            n -= d* num;
            d++;
             num *= 10;
        }
        // 算出第几个数
        int v = (n - 1) / d + 1;
        int cur = n - (v - 1) * d;
        int k = (int)(num / 9) + v - 1;
        return (int)(k / Math.pow(10, cur)) % 10;
    }

    public int largestSumAfterKNegations(int[] nums, int k) {
        PriorityQueue<Integer> queue = new PriorityQueue<>((o1, o2) -> o2 - o1);
        int ans = 0;
        for (int i = 0;i < nums.length; i++) {
            queue.add(nums[i]);
        }
        while (k-- > 0) {
            queue.add(queue.poll() * -1);
        }
        while (!queue.isEmpty()) {
            ans += queue.poll();
        }
        return ans;
    }

    public int qpow(int a, int b, int mod) {
        int ans = 1;
        a %= mod;
        while (b > 0) {
            if ((b & 1) == 1) {
                ans = ans * a % mod;
            }
            a = a * a % mod;
            b >>= 1;
        }
        return ans;
    }

    public int superPow(int a, int[] b) {
        int ans = 1;
        for (int i = 0;i < b.length; i++) {
            ans = qpow(ans, 10, 1337) * qpow(a, b[i], 1337);
            ans %= 1337;
        }
        return ans;
    }

    public int[] findEvenNumbers(int[] digits) {
        Arrays.sort(digits);
        List<Integer> list = new ArrayList<>();
        Set<Integer> set = new HashSet<>();
        for (int i = 0;i < digits.length; i++) {
            if (digits[i] == 0) {
                continue;
            }
            StringBuilder sb = new StringBuilder();
            sb.append(digits[i]);
            for (int j = 0;j < digits.length; j++) {
                if (i != j) {
                    sb.append(digits[j]);
                    for (int z = 0;z < digits.length; z++) {
                        if (z != j && z != i) {
                            sb.append(digits[z]);
                            if (sb.length() == 3) {
                                Integer cur = Integer.parseInt(sb.toString());
                                if (cur % 2 == 0 && !set.contains(cur)) {
                                    list.add(cur);
                                    set.add(cur);
                                }
                            }
                            sb.deleteCharAt(sb.length() - 1);
                        }
                    }
                    sb.deleteCharAt(sb.length() - 1);
                }
            }
        }
        int[] ans = new int[list.size()];
        for (int i = 0;i < list.size(); i++) {
            ans[i] = list.get(i);
        }
        return ans;
    }

    public ListNode deleteMiddle(ListNode head) {
        ListNode hair = new ListNode(0);
        hair.next = head;
        int n = 0;
        ListNode p = head;
        while (p != null) {
            p = p.next;
            n++;
        }
        int cur = n % 2 == 1 ? n / 2 : (n / 2 + 1);
        ListNode q = head;
        p = hair;
        n = 0;
        while (n < cur) {
            p = p.next;
            q = q.next;
            n++;
        }
        p.next = q.next;
        return hair.next;
    }

    public String getDirections(TreeNode root, int startValue, int destValue) {
        Map<Integer, List<Pair>> map = new HashMap<>();
        dfs(root, map);
        Queue<Integer> queue = new ArrayDeque<>();
        Pair[] pairs = new Pair[map.size() + 1];
        int[] vis = new int[map.size() + 1];
        queue.add(startValue);
        while (!queue.isEmpty()) {
            int s = queue.poll();
            if (s == destValue) {
                break;
            }
            if (vis[s] == 1) {
                continue;
            }
            vis[s] = 1;
            for (Pair pair : map.get(s)) {
                if (vis[pair.x] == 0) {
                    queue.add(pair.x);
                    pairs[pair.x] = new Pair(s, pair.y);
                }
            }
        }
        int cur = destValue;
        StringBuilder sb = new StringBuilder();
        do {
            if (pairs[cur].y == 1) {
                sb.append("L");
            } else if (pairs[cur].y == 2) {
                sb.append("R");
            } else {
                sb.append("U");
            }
            cur = pairs[cur].x;
        } while (pairs[cur] != null);
        System.out.println(sb.toString());
        return sb.reverse().toString();
    }

    private void dfs(TreeNode root, Map<Integer, List<Pair>> map) {
        if (root.left != null) {
            map.putIfAbsent(root.val, new ArrayList<>());
            map.putIfAbsent(root.left.val, new ArrayList<>());
            map.get(root.val).add(new Pair(root.left.val, 1));
            map.get(root.left.val).add(new Pair(root.val, 3));
            dfs(root.left, map);
        }
        if (root.right != null) {
            map.putIfAbsent(root.val, new ArrayList<>());
            map.putIfAbsent(root.right.val, new ArrayList<>());
            map.get(root.val).add(new Pair(root.right.val, 2));
            map.get(root.right.val).add(new Pair(root.val, 3));
            dfs(root.right, map);
        }
    }

    public int[] maxSubsequence(int[] nums, int k) {
        int[] ans = new int[k];
        Pair[] pairs = new Pair[nums.length];
        Pair[] newPairs = new Pair[k];
        int cur = 0;
        for (int i = nums.length - 1;i >= 0; i--) {
            pairs[cur++] = new Pair(nums[i], i);
        }
        Arrays.sort(pairs, (o1, o2) -> o2.x - o1.x);
        for (int i = 0;i < nums.length; i++) {
            newPairs[cur++] = new Pair(pairs[i].x, pairs[i].y);
            if (cur == k) {
                break;
            }
        }
        Arrays.sort(newPairs, (o1, o2) -> o2.y - o1.y);
        for (int i = 0;i <k; i++) {
            ans[i] = newPairs[i].x;
        }
        return ans;
    }

    public List<Integer> goodDaysToRobBank(int[] security, int time) {
        List<Integer> list = new ArrayList<>();
        int n = security.length;
        if (time * 2 + 1 > n) {
            return new ArrayList<>();
        }
        if (time == 0) {
            for (int i = 0;i < n; i++) {
                list.add(i);
            }
        } else {
            int[] l = new int[n];
            int[] r = new int[n];
            Arrays.fill(l, 1);
            Arrays.fill(r, 1);
            for (int i = 1;i < n; i++) {
                l[i] = l[i - 1];
                if (security[i] <= security[i - 1]) {
                    l[i]++;
                } else {
                    l[i] = 1;
                }
            }
            for (int i = n - 2;i >= 0; i--) {
                r[i] = r[i + 1];
                if (security[i] <= security[i + 1]) {
                    r[i]++;
                } else {
                    r[i] = 1;
                }
            }
            for (int i = 0;i < security.length; i++) {
                if (l[i] - 1 >= time && r[i] - 1 >= time) {
                    list.add(i);
                }
            }
        }
        return list;
    }

    public int maximumDetonation(int[][] bombs) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        int n = bombs.length;
        for (int i = 0;i < n; i++) {
            for (int j = 0;j < n; j++) {
                if (i != j) {
                    if ((long) (bombs[i][0] - bombs[j][0]) * (bombs[i][0] - bombs[j][0]) +
                            (long) (bombs[i][1] - bombs[j][1]) * (bombs[i][1] - bombs[j][1]) <= (long) bombs[i][2] * bombs[i][2]) {
                        map.putIfAbsent(i, new ArrayList<>());
                        map.get(i).add(j);
                    }
                }
            }
        }
        int ans = 0;
        for (int i = 0;i < n; i++) {
            int[] vis = new int[n];
            Queue<Integer> que = new ArrayDeque<>();
            que.add(i);
            int cur = 0;
            while (!que.isEmpty()) {
                int p = que.poll();
                if (vis[p] == 1) {
                    continue;
                }
                vis[i] = 1;
                cur++;
                if (map.get(p) == null) {
                    continue;
                }
                for (int v : map.get(p)) {
                    if (vis[v] == 1) {
                        continue;
                    }
                    que.add(v);
                }
            }
            ans = Math.max(cur, ans);
        }
        return ans;
    }

    public int countPoints(String rings) {
        Map<Integer, Set<Character>> map = new HashMap<>();
        int ans = 0;
        for (int i = 0;i < rings.length(); i+=2) {
            Integer key = Integer.parseInt(String.valueOf(rings.charAt(i+1)));
            map.putIfAbsent(key, new HashSet<>());
            map.get(key).add(rings.charAt(i));
        }
        for (int key : map.keySet()) {
            if (map.get(key).size() == 3) {
                ans++;
            }
        }
        return ans;
    }

    public long subArrayRanges(int[] nums) {
        long ans = 0;
        for (int i = 0;i < nums.length; i++) {
            int min = nums[i];
            int max = nums[i];
            for (int j = i + 1;j < nums.length; j++) {
                min = Math.min(nums[j], min);
                max = Math.max(nums[j], max);
                ans += max - min;
            }
        }
        return ans;
    }

    public int minimumRefill(int[] plants, int capacityA, int capacityB) {
        int l = 0, r = plants.length - 1;
        int curA = capacityA;
        int curB = capacityB;
        int ans = 0;
        while (l <= r) {
            if (l == r) {
                if (curA < plants[l] && curB < plants[l]) {
                    ans++;
                    break;
                }
            }
            if (curA >= plants[l]) {
                curA -= plants[l];
            } else {
                curA = capacityA;
                curA -= plants[l];
                ans++;
            }
            if (curB >= plants[r]) {
                curB -= plants[r];
            } else {
                curB = capacityB;
                curB -= plants[r];
                ans++;
            }
            l++;r--;
        }
        return ans;
    }

    public int maxTotalFruits(int[][] fruits, int startPos, int k) {
        int ans = 0;
        int[] cur = new int[101010];
        int[] sum = new int[101010];
        for (int i = 0;i < fruits.length; i++) {
            cur[fruits[i][0]] = fruits[i][1];
        }
        for (int i = 1;i < sum.length; i++) {
            if (i == 0) {
                sum[i] = cur[i - 1];
            } else {
                sum[i] = sum[i - 1] + cur[i - 1];
            }
        }
        // 向左走i步
        for (int i = 0;i < k; i++) {
            int l = Math.max(0, startPos - i);
            int r = Math.min(101010, l + (k - i));
            int curVal = sum[r] - sum[l - 1];
            ans = Math.max(ans, curVal);
        }
        // 向右走i步
        for (int i = 0;i < k; i++) {
            int r = Math.min(101010, startPos + i);
            int l = Math.max(0, r - (k - i));
            int curVal = sum[r] - sum[l];
            ans = Math.max(ans, curVal);
        }
        return ans;
    }

    public int scheduleCourse(int[][] courses) {
        Arrays.sort(courses, (a, b) -> (a[1] - b[1]));
        Queue<Integer> que = new PriorityQueue<>((a, b) -> (b - a));
        int total = 0;
        for (int[] c : courses) {
            int ti = c[0], di = c[1];
            if (total + ti <= di) {
                total += ti;
                que.offer(ti);
            } else if (!que.isEmpty() && que.peek() > ti) {
                total -= que.poll() - ti;
                que.offer(ti);
            }
        }
        return que.size();
    }

    public int[] loudAndRich(int[][] richer, int[] quiet) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0;i < richer.length; i++) {
            map.putIfAbsent(richer[i][1], new ArrayList<>());
            map.get(richer[i][1]).add(richer[i][0]);
        }
        int[] ans = new int[quiet.length];
        for (int i = 0;i < quiet.length; i++) {
            int cur = Integer.MAX_VALUE;
            int curVal = 0;
            Queue<Integer> que = new LinkedList<>();
            que.add(i);
            int[] vis = new int[quiet.length];
            while (!que.isEmpty()) {
                int v = que.poll();
                if (vis[v] == 1) {
                    continue;
                }
                vis[v] = 1;
                if (map.get(v) == null) {
                    continue;
                }
                for (int x : map.get(v)) {
                    if (quiet[x] < cur) {
                        cur = quiet[x];
                        curVal = x;
                    }
                    que.add(x);
                }
            }
            ans[i] = curVal;
        }
        return ans;
    }

    public int countBattleships(char[][] board) {
        int ans = 0;
        int[][] vis = new int[board.length][board[0].length];
        for (int i = 0;i < board.length; i++) {
            for (int j = 0;j < board[0].length; j++) {
                if (board[i][j] == 'X' && vis[i][j] == 0) {
                    dfs(board, i, j, vis);
                    ans++;
                }
            }
        }
        return ans;
    }

    public void dfs(char[][] board, int x, int y, int[][] vis) {
        if (x < 0 || x >= board[0].length || y < 0 || y >= board.length) {
            return ;
        }
        vis[x][y] = 1;
        dfs(board, x + 1, y, vis);
        dfs(board, x - 1, y, vis);
        dfs(board, x, y - 1, vis);
        dfs(board, x, y + 1, vis);
    }

    public String firstPalindrome(String[] words) {
        for (int i = 0;i <words.length; i++) {
            boolean flag = true;
            int n = words[i].length();
            for (int j = 0;j < n / 2; j++) {
                if (words[i].charAt(j) != words[i].charAt(n - 1 - j)) {
                    flag = false;
                }
            }
            if (flag) {
                return words[i];
            }
        }
        return "";
    }

    public String addSpaces(String s, int[] spaces) {
        StringBuilder sb = new StringBuilder();
        int cur = 0;
        for (int i = 0;i < s.length(); i++) {
            if (cur >= spaces.length) {
                sb.append(s.charAt(i));
                continue;
            }
            if (i != spaces[cur]) {
                sb.append(s.charAt(i));
            } else {
                sb.append(" ");
                sb.append(s.charAt(i));
                cur++;
            }
        }
        return  sb.toString();
    }

    public long getDescentPeriods(int[] prices) {
        long ans = prices.length;
        long cur = 1;
        for (int i = 1;i < prices.length; i++) {
            if (prices[i - 1] - prices[i] == 1) {
                cur++;
            } else {
                ans += ((cur * (cur + 1)) / 2 - cur);
                cur = 1;
            }
        }
        if (cur != 1) {
            ans += ((cur * (cur + 1)) / 2 - cur);
        }
        return ans;
    }

    public int kIncreasing(int[] arr, int k) {
        int ans = 0;
        List<List<Integer>> list = new ArrayList<>();
        for (int i = 0;i < k; i++) {
            int v = i;
            List<Integer> curList = new ArrayList<>();
            while (v < arr.length) {
                curList.add(arr[v]);
                vis[v] = 1;
                v += k;
            }
            list.add(curList);
        }
        for (int i = 0;i < list.size(); i++) {
            int[] dp = new int[list.get(i).size()];
            dp[0] = list.get(i).get(0);
            int len = 0;
            for (int num : list.get(i)) {
                int l = 0,r = len;
                while (l < r) {
                    int mid = (l + r) >> 1;
                    if (dp[mid] <= num) {
                        l = mid + 1;
                    } else {
                        r = mid;
                    }
                }
                dp[l] = num;
                if (len == r) {
                    len++;
                }
            }
            ans += list.get(i).size() - len;
        }
        return ans;
    }

    public int findRadius(int[] houses, int[] heaters) {
        int ans = 0;
        Arrays.sort(houses);
        Arrays.sort(heaters);
        for (int house : houses) {
            int i = binarySearch(heaters, house);
            int j = i - 1;
            int curA = j < 0 ? Integer.MAX_VALUE : Math.abs(house - heaters[j]);
            int curB = i >= heaters.length ? Integer.MAX_VALUE : Math.abs(heaters[i] - house);
            ans = Math.max(ans, Math.min(curA, curB));
        }
        return ans;
    }

    public int binarySearch(int[] nums, int tar) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int mid = (l + r) >> 1;
            if (nums[mid] >= tar) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    public int dayOfYear(String date) {
        String[] strings = date.split("-");
        int[] mod = {0,31,28,31,30,31,30,31,31,30,31,30,31};
        int year = Integer.parseInt(strings[0]);
        if (year % 400 == 0 || (year % 4 == 0 && year % 100 != 0)) {
            mod[2] = 29;
        }
        int month = Integer.parseInt(strings[1]);
        int ans = Integer.parseInt(strings[2]);
        for (int i = 1;i <= month - 1; i++) {
            ans += mod[i];
        }
        return ans;
    }

    public int repeatedStringMatch(String a, String b) {
        int len = a.length() * 2 + b.length();
        int cnt = 1;
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(a);
        while (cnt * a.length() < len) {
            if (stringBuilder.toString().contains(b)) {
                break;
            }
            cnt++;
            stringBuilder.append(a);
        }
        return cnt;
    }

    int N = (int)1e5 + 10, P = 6551;
    int[] hash = new int[N], p = new int[N];

    public List<String> findRepeatedDnaSequences(String s) {
        int n = s.length();
        List<String> ans = new ArrayList<>();
        p[0] = 1;
        for (int i = 1;i <= n; i++) {
            hash[i] = hash[i - 1] * P + s.charAt(i - 1);
            p[i] = p[i - 1] * P;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 1;i + 10 - 1 <= n; i++) {
            int j = i + 10 - 1;
            int h = hash[j] - hash[i - 1] * p[j - i + 1];
            int cnt = map.getOrDefault(h, 0);
            if (cnt == 1) {
                ans.add(s.substring(i - 1, i + 10 - 1));
            }
            map.put(h, cnt + 1);
        }
        return ans;
    }

    public int eatenApples(int[] apples, int[] days) {
        int ans = 0;
        Queue<int[]> que = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        int n = apples.length;
        int i = 0;
        while (i < n) {
            while (!que.isEmpty() && que.peek()[0] <= i) {
                que.poll();
            }
            int ex = i + days[i];
            int count = apples[i];
            if (count > 0) {
                que.offer(new int[] {ex, count});
            }
            if (!que.isEmpty()) {
                int[] curArr = que.peek();
                curArr[1]--;
                if (curArr[1] == 0) {
                    que.poll();
                }
                ans++;
            }
            i++;
        }
        while (!que.isEmpty()) {
            while (!que.isEmpty() && que.peek()[0] <= i) {
                que.poll();
            }
            if (que.isEmpty()) {
                break;
            }
            int[] curArr = que.poll();
            int cur = Math.min(curArr[0] - i, curArr[1]);
            ans += cur;
            i += cur;
        }
        return ans;
    }

    public boolean isEvenOddTree(TreeNode root) {
        Queue<TreeNode> que = new LinkedList<>();
        que.add(root);
        int level = 0;
        while (!que.isEmpty()) {
            int size = que.size();
            int last = que.peek().val;
            for (int i = 1; i <= size; i++) {
                TreeNode curNode = que.poll();
                int v = curNode.val;
                if (level % 2 == v % 2) {
                    return false;
                }
                if (i != 1 && level % 2 == 0 && v <= last) {
                    return false;
                }
                if (i != 1 && level % 2 != 0 && v >= last) {
                    return false;
                }
                if (curNode.left != null) {
                    que.add(curNode.left);
                }
                if (curNode.right != null) {
                    que.add(curNode.right);
                }
            }
            level++;
        }
        return true;
    }

    public int mostWordsFound(String[] sentences) {
        int ans = 0;
        for (int i = 0;i < sentences.length; i++) {
            String[] strings = sentences[i].split(" ");
            ans = Math.max(ans, strings.length);
        }
        return ans;
    }

    public List<String> findAllRecipes(String[] recipes, List<List<String>> ingredients, String[] supplies) {
        Set<String> set = new HashSet<>(Arrays.asList(supplies));
        List<String> ans = new ArrayList<>();
        while (true) {
            boolean found = false;
            for (int i = 0;i < recipes.length; i++) {
                if (set.containsAll(ingredients.get(i))) {
                    ans.add(recipes[i]);
                    set.add(recipes[i]);
                    found = true;
                }
            }
            if (!found) {
                break;
            }
        }
        return ans;
    }

    public String[] findOcurrences(String text, String first, String second) {
        String[] texts = text.split(" ");
        List<String> list = new ArrayList<>();
        for (int i = 0;i < texts.length - 2; i++) {
            if (texts[i].equals(first) && texts[i + 1].equals(second)) {
                list.add(texts[i + 2]);
            }
        }
        String[] ans = new String[list.size()];
        for (int i = 0;i < list.size(); i++) {
            ans[i] = list.get(i);
        }
        return ans;
    }

    public boolean isSameAfterReversals(int num) {
        String s = String.valueOf(num);
        StringBuilder sb = new StringBuilder();
        sb.append(s);
        sb.reverse();
        int vv = Integer.parseInt(sb.toString());
        StringBuilder sb1 = new StringBuilder();
        sb1.append(vv);
        sb1.reverse();
        return num == Integer.parseInt(sb1.toString());
    }

    public int[] executeInstructions(int n, int[] startPos, String s) {
        int[] ans = new int[s.length()];
        for (int i = 0;i < s.length(); i++) {
            String newS = s.substring(i);
            int cur = 0;
            int nx = startPos[0];
            int ny = startPos[1];
            while (cur < newS.length()) {
                char x = newS.charAt(cur);
                if (x == 'L') {
                    ny -= 1;
                } else if (x == 'R') {
                    ny += 1;
                } else if (x == 'D') {
                    nx += 1;
                } else if (x == 'U') {
                    nx -= 1;
                }
                if (nx >= n || ny >= n || nx < 0 || ny < 0) {
                    break;
                }
                cur++;
            }
            ans[i] = cur;
        }
        return ans;
    }

    public long[] getDistances(int[] arr) {
        long[] ans = new long[arr.length];
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0;i < arr.length; i++) {
            map.putIfAbsent(arr[i], new ArrayList<>());
            map.get(arr[i]).add(i);
        }
        for (int x : map.keySet()) {
            List<Integer> integers = map.get(x);
            int n = integers.size();
            if (n == 1) {
                ans[integers.get(0)] = 0;
                continue;
            }
            long v = 0;
            for (int i = 1;i < n; i++) {
                long vv = integers.get(i) - integers.get(i - 1);
                v += i * vv;
                ans[integers.get(i)] += v;
            }
            v = 0;
            int a = 1;
            for (int i = n-2;i >= 0; i--) {
                long vv = integers.get(i+1) - integers.get(i);
                v += (a++) * vv;
                ans[integers.get(i)] += v;
            }
        }
        return ans;
    }

    public int[] recoverArray(int[] nums) {
        Arrays.sort(nums);
        int cur = nums.length >> 1;
        int[] ans = new int[cur];
        for (int i = 1;i < nums.length; i++) {
            int vv = (nums[i] - nums[0]);
            if (vv == 0) {
                continue;
            }
            if (vv % 2 == 0) {
                int a = 0;
                List<Integer> set = new ArrayList<>();
                for (int k : nums) {
                    set.add(k);
                }
                for (int j = 0;j < nums.length; j++) {
                    int num = nums[j];
                    if (set.contains(num + vv) && set.contains(num)) {
                        int i1 = Collections.binarySearch(set, num + vv);
                        ans[a++] = num + vv / 2;
                        set.remove(j);
                        set.remove(i1);
                    }
                    if (a == cur) {
                        break;
                    }
                }
                if (a == cur) {
                    break;
                }
            }
        }
        return ans;
    }

    public int numberOfBeams(String[] bank) {
        int ans = 0;
        int last = 0;
        for (int i = 0;i < bank.length; i++) {
            int cur = 0;
            for (int j = 0;j < bank[i].length(); j++) {
                if (bank[i].charAt(j) == '1') {
                    cur++;
                }
            }
            if (cur != 0) {
                ans += (last * cur);
                last = cur;
            }
        }
        return ans;
    }

    public boolean asteroidsDestroyed(int mass, int[] asteroids) {
        Arrays.sort(asteroids);
        for (int i = 0;i < asteroids.length; i++) {
            if (mass < asteroids[i]) {
                return false;
            } else {
                mass += asteroids[i];
            }
        }
        return true;
    }

    public int maximumInvitations(int[] favorite) {
        int n = favorite.length;
        Map<Integer, List<Integer>> rg = new HashMap<>();
        int[] deg = new int[n];
        for (int v = 0;v < n; v++) {
            int w = favorite[v];
            rg.putIfAbsent(w, new ArrayList<>());
            rg.get(w).add(v);
            deg[w]++;
        }
        // 拓扑排序
        Queue<Integer> que = new LinkedList<>();
        for (int i = 0;i < n; i++) {
            if (deg[i] == 0) {
                que.add(i);
            }
        }
        while (!que.isEmpty()) {
            int v = que.poll();
            int w = favorite[v];
            if (--deg[w] == 0) {
                que.add(w);
            }
        }
        int maxRingSize = 0, sumChainSize = 0;
        for (int i = 0;i < n; i++) {
            if (deg[i] <= 0) {
                continue;
            }
            deg[i] = -1;
            int curSize = 1;
            for (int v = favorite[i];v != i; v = favorite[v]) {
                deg[v] = -1;
                curSize++;
            }
            if (curSize == 2) {
                sumChainSize = dfsRg(rg, i, deg) + dfsRg(rg, favorite[i], deg);
            } else {
                maxRingSize = Math.max(curSize, maxRingSize);
            }
        }
        return Math.max(maxRingSize, sumChainSize);
    }

    public int dfsRg(Map<Integer, List<Integer>> rg, int v, int[] deg) {
        if (rg.get(v) == null) {
            return 1;
        }
        int cur = 0;
        for (int w : rg.get(v)) {
            if (deg[w] == 0) {
                cur += Math.max(cur, dfsRg(rg, w, deg) + 1);
            }
        }
        return cur;
    }

    public String simplifyPath(String path) {
        String[] names = path.split("/");
        Deque<String> stack = new ArrayDeque<>();
        for (String name : names) {
            if ("..".equals(name)) {
                if (!stack.isEmpty()) {
                    stack.pollLast();
                }
            } else if (name.length() > 0 && !".".equals(name)) {
                stack.offerLast(name);
            }
        }
        StringBuilder sb = new StringBuilder();
        if (stack.isEmpty()) {
            sb.append("/");
        } else {
            while (!stack.isEmpty()) {
                sb.append("/");
                sb.append(stack.pollFirst());
            }
        }
        return sb.toString();
    }

    public int maxDepth(String s) {
        int ans = 0;
        int cur = 0;
        for (int i = 0;i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                cur++;
                ans = Math.max(cur, ans);
            } else if (s.charAt(i) == ')') {
                cur--;
            }
        }
        return ans;
    }

    public String capitalizeTitle(String title) {
        String[] strings = title.split(" ");
        StringBuilder sb = new StringBuilder();
        for (int i = 0;i < strings.length; i++) {
            if (strings[i].length() == 1 || strings[i].length() == 2) {
                sb.append(strings[i].toLowerCase());
            } else {
                String s1 = strings[i].toLowerCase();
                char[] chars = s1.toCharArray();
                if (chars[0] >= 'a' && chars[0] <= 'z') {
                    chars[0] = (char) (chars[0] - 32);
                }
                sb.append(new String(chars));
            }
            if (i != strings.length - 1) {
                sb.append(" ");
            }
        }
        return sb.toString();
    }

    public int pairSum(ListNode head) {
        List<Integer> list = new ArrayList<>();
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }
        int n = list.size();
        int ans = 0;
        for (int i = 0;i < n / 2; i++) {
            ans = Math.max(ans, list.get(i) + list.get(n - i - 1));
        }
        return ans;
    }

    public int longestPalindrome(String[] words) {
        Map<String, Integer> map = new HashMap<>();
        int ans = 0;
        for (int i = 0;i < words.length; i++) {
            map.put(words[i], map.getOrDefault(words[i], 0) + 1);
        }
        for (String s : map.keySet()) {
            if (map.get(s) > 0) {
                String s1 = new StringBuilder(s).reverse().toString();
                if (s1.equals(s)) {
                    int cur = map.get(s);
                    ans += (cur / 2) * 2;
                    map.put(s, cur - (cur / 2));
                } else {
                    int cur1 = map.get(s1) == null ? 0 : map.get(s1);
                    int cur2 = map.get(s);
                    int vv = Math.min(cur1, cur2);
                    ans += vv * 2;
                    if (map.get(s1) != null) {
                        map.put(s1, cur1 - vv);
                    }
                    map.put(s, cur2 - vv);
                }
            }
        }
        for (String s : map.keySet()) {
            if (map.get(s) > 0) {
                if (s.charAt(0) == s.charAt(1)) {
                    ans += 2;
                    break;
                }
            }
        }
        return ans;
    }

    public boolean checkValid(int[][] matrix) {
        int vv = 0;
        int n = matrix.length;
        for (int i = 0;i < n; i++) {
            int[] vis1 = new int[n + 1];
            int[] vis2 = new int[n + 1];
            for (int j = 0;j < n; j++) {
                if (vis1[matrix[i][j]] != 0) {
                    return false;
                }
                if (vis1[matrix[j][i]] != 0) {
                    return false;
                }
                vis1[matrix[i][j]] = 1;
                vis1[matrix[j][i]] = 1;
            }
        }
        return true;
    }

    public int minSwaps(int[] nums) {
        int cur = 0;
        boolean flag = true;
        for (int i = 0;i < nums.length; i++) {
            if (nums[i] == 1) {
                cur++;
            } else {
                flag = false;
            }
        }
        if (flag) {
            return 0;
        }
        int j = 0;
        int count = 0;
        while (j < cur) {
            if (nums[j++] == 1) {
                count++;
            }
        }
        int minsp = Integer.MAX_VALUE;
        for (int i = 0;j < nums.length + cur - 1; j++) {
            if (nums[j % nums.length] == 1) {
                count++;
            }
            minsp = Math.min(minsp, cur - count);
            if (nums[i++] == 1) {
                count--;
            }
        }
        return minsp;
    }

    public int wordCount(String[] startWords, String[] targetWords) {
        Set<Integer> set = new HashSet<>();
        int ans = 0;
        for (int i = 0;i < startWords.length; i++) {
            int cur = 0;
            for (int j = 0;j < startWords[i].length(); j++) {
                cur |= (1 << (startWords[i].charAt(j) - 'a'));
            }
            set.add(cur);
        }
        for (int i = 0;i < targetWords.length; i++) {
            int cur = 0;
            for (int j = 0;j < targetWords[i].length(); j++) {
                cur |= (1 << (targetWords[i].charAt(j) - 'a'));
            }
            boolean ok = false;
            for (int j = 0;j < 26; j++) {
                if ((cur & (1 << j)) > 0) {
                    if (set.contains(cur - (1 << j))) {
                        ok = true;
                        break;
                    }
                }
            }
            if (ok) {
                ans++;
            }
        }
        return ans;
    }

    public int earliestFullBloom(int[] p, int[] g) {
        int n = p.length;
        int[][] info = new int[n][2];
        for (int i = 0;i < n; i++) {
            info[i][0] = p[i];
            info[i][1] = g[i];
        }
        Arrays.sort(info, (i1, i2) -> i2[1] - i1[1]);
        int ans = 0;
        int day = 0;
        for (int[] v : info) {
            day += v[0];
            ans = Math.max(ans, v[1] + day);
        }
        return ans;
    }

    public boolean isAdditiveNumber(String num) {
        int n = num.length();
        for (int secondStart = 1;secondStart < n;  secondStart++) {
            if (num.charAt(0) == '0' && secondStart != 1) {
                break;
            }
            for (int secondEnd = secondStart;secondEnd < n - 1; secondEnd++) {
                if (num.charAt(secondStart) == '0' && secondEnd != secondStart){
                    break;
                }
                if (ok(secondStart, secondEnd, num)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean ok(int secondStart, int secondEnd, String num) {
        int n = num.length();
        int s = 0, e = secondStart - 1;
        while (secondEnd <= n - 1) {
            String third = stringAdd(num, s, e, secondStart, secondEnd);
            int ts = secondEnd + 1;
            int te = ts + third.length() - 1;
            if (te >= n || !num.substring(ts, te + 1).equals(third)) {
                break;
            }
            if (te == n - 1) {
                return true;
            }
            s = secondStart;
            e = secondEnd;
            secondEnd = te;
            secondStart = ts;
        }
        return false;
    }

    public String stringAdd(String s, int firstStart, int firstEnd, int secondStart, int secondEnd) {
        StringBuffer third = new StringBuffer();
        int carry = 0, cur = 0;
        while (firstEnd >= firstStart || secondEnd >= secondStart || carry != 0) {
            cur = carry;
            if (firstEnd >= firstStart) {
                cur += s.charAt(firstEnd) - '0';
                --firstEnd;
            }
            if (secondEnd >= secondStart) {
                cur += s.charAt(secondEnd) - '0';
                --secondEnd;
            }
            carry = cur / 10;
            cur %= 10;
            third.append((char) (cur + '0'));
        }
        third.reverse();
        return third.toString();
    }

    public boolean increasingTriplet(int[] nums) {
        if (nums.length < 3) {
            return false;
        }
        int cur1 = nums[0];
        int cur2 = Integer.MAX_VALUE;
        for (int i = 1;i < nums.length; i++) {
            int cur = nums[i];
            if (cur > cur2) {
                return true;
            }
            if (cur > cur1) {
                cur2 = cur;
            } else if (cur < cur1) {
                cur1 = cur;
            }
        }
        return false;
    }

    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> list = new LinkedList<>();
        PriorityQueue<int[]> que = new PriorityQueue<>((o1, o2) -> (o2[0] + o2[1]) - (o1[0] + o1[1]));
        for (int i = 0;i < nums1.length && i < k; i++) {
            for (int j = 0;j < nums2.length && j < k; j++) {
                if (que.size() < k) {
                    que.add(new int[]{nums1[i], nums2[j]});
                } else if (que.peek()[0] + que.peek()[1] > nums1[i] + nums2[j]) {
                    que.poll();
                    que.add(new int[]{nums1[i], nums2[j]});
                }
            }
        }
        while (!que.isEmpty()) {
            List<Integer> l1 = new ArrayList<>();
            l1.add(que.peek()[0]);
            l1.add(que.peek()[1]);
            list.add(0, l1);
            que.poll();
        }
        return list;
    }

    public String[] divideString(String s, int k, char fill) {
        int len = s.length() % k == 0 ? s.length() / k : s.length() / k + 1;
        String[] strings = new String[len];
        for (int i = 0;i < len; i++) {
            int cur = 0;
            StringBuilder sb = new StringBuilder();
            while (cur < k) {
                int vv = i * k + cur;
                if (vv < s.length()) {
                    sb.append(s.charAt(cur));
                } else {
                    sb.append(fill);
                }
                cur++;
            }
            strings[i] = sb.toString();
        }
        return strings;
    }

    public int minMoves(int target, int maxDoubles) {
        int ans = 0;
        while (target != 1) {
            if (target % 2 == 0 && maxDoubles != 0) {
                target /= 2;
                ans++;
                maxDoubles--;
            } else if (target % 2 != 0) {
                target--;
                ans++;
            }
            if (maxDoubles == 0) {
                break;
            }
        }
        ans += target - 1;
        return ans;
    }

    public long mostPoints(int[][] questions) {
        int n = questions.length;
        long[] f = new long[n + 1];
        for (int i = n - 1;i >= 0; --i) {
            int[] question = questions[i];
            int j = i + question[1] + 1;
            if (j >= n) {
                j = n;
            }
            f[i] = Math.max(f[i + 1], question[0] + f[j]);
        }
        return f[0];
    }

    public int countVowelPermutation(int n) {
        long  mod = (long ) 1e9;
        long [] dp = new long [5];
        long [] ndp = new long [5];
        Arrays.fill(dp, 1);
        for (int i = 0;i < n; i++) {
            ndp[0] = (dp[1] + dp[2] + dp[4]) % mod;
            ndp[1] = (dp[0] + dp[2]) %mod;
            ndp[2] = (dp[1] + dp[3]) % mod;
            ndp[3] = (dp[2]) % mod;
            ndp[4] = (dp[2] + dp[3]) % mod;
            System.arraycopy(ndp, 0, dp, 0, 5);
        }
        long ans = 0;
        for (int i = 0;i < 5; i++) {
            ans = (ans + dp[i]) % mod;
        }
        return (int) ans;
    }

    public int findMinDifference(List<String> timePoints) {
        Collections.sort(timePoints);
        int ans = Integer.MAX_VALUE;
        for (int i = 0;i < timePoints.size(); i++) {
            int v2 = i == 0 ? timePoints.size() - 1 : i - 1;
            String s1 = timePoints.get(v2);
            String s2 = timePoints.get(i);
            String[] strings1 = s1.split(":");
            String[] strings2 = s2.split(":");
            int cur1 = Integer.parseInt(strings2[0]) * 60 + Integer.parseInt(strings2[1]);
            int cur2 = Integer.parseInt(strings1[0]) * 60 + Integer.parseInt(strings1[1]);
            ans = Math.min(ans, Math.abs(cur1 - cur2));
            ans = Math.min(ans, 1400 - Math.abs(cur1 - cur2));
        }
        return ans;
    }

    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0;i < nums.length; i++) {
            if (map.containsKey(nums[i]) && i - map.get(nums[i]) <= k) {
                return true;
            }
            map.put(nums[i], i);
        }
        return false;
    }

    public int minimumCost(int[] cost) {
        int ans = 0;
        Arrays.sort(cost);
        for (int i = cost.length - 1;i >= 0; i-= 3) {
            if (i - 1 >= 0) {
                ans += cost[i] + cost[i - 1];
            } else {
                ans += cost[i];
            }
        }
        return ans;
    }

    public int numberOfArrays(int[] differences, int lower, int upper) {
        long[] curArr = new long[differences.length + 1];
        curArr[0] = 1;
        long max = curArr[0];
        long min = curArr[0];
        for (int i = 0;i < differences.length; i++) {
            curArr[i + 1] = curArr[i] + differences[i];
            max = Math.max(max, curArr[i + 1]);
            min = Math.min(min, curArr[i + 1]);
        }
        if (max - min > upper - lower) {
            return 0;
        }
        if (max > upper) {
            min = min - (max - upper);
            return (int) (min - lower + 1);
        } else if (min < lower) {
            max = max + (lower - min);
            return (int) (upper - max + 1);
        } else {
            return (int) ((upper - max) + (min - lower) + 1);
        }
    }

    public List<List<Integer>> highestRankedKItems(int[][] grid, int[] pricing, int[] start, int k) {
        int[][] dir = new int[][] {
                {0,-1},{1,0},{0,1},{-1,0}
        };
        int n = grid.length;
        int m = grid[0].length;
        List<int[]> list = new ArrayList<>();
        Queue<int[]> que = new ArrayDeque<>();
        int[][] vis = new int[grid.length][grid[0].length];
        que.add(new int[]{start[0], start[1], 0});
        while (!que.isEmpty()) {
            int[] curPos = que.poll();
            int x = curPos[0];
            int y = curPos[1];
            if (vis[x][y] == 1) {
                continue;
            }
            vis[x][y] = 1;
            if (grid[x][y] >= pricing[0] && grid[x][y] <= pricing[1]) {
                list.add(new int[]{x, y, curPos[2], grid[x][y]});
            }
            for (int i = 0;i < 4; i++) {
                int nx = x + dir[i][0];
                int ny = y + dir[i][1];
                if (nx < 0 || nx >= n || ny <0 || ny >= m) {
                    continue;
                }
                if (grid[nx][ny] == 0) {
                    continue;
                }
                que.add(new int[]{nx, ny, curPos[2] + 1});
            }
        }
        list.sort((o1, o2) -> {
            if (o1[2] != o2[2]) {
                return o1[2] - o2[2];
            }
            if (o1[3] != o2[3]) {
                return o1[3] - o2[3];
            }
            if (o1[0] != o2[0]) {
                return o1[0] - o2[0];
            }
            return o1[1] - o2[1];
        });
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0;i < k && i < list.size(); i++) {
            List<Integer> curList = new ArrayList<>();
            int[] vv = list.get(i);
            curList.add(vv[0]);
            curList.add(vv[1]);
            ans.add(curList);
        }
        return ans;
    }

    public int numberOfWays(String s) {
        List<Integer> list = new ArrayList<>();
        int mod = (int) (1e9 + 7);
        int cur = 0;
        int vv = 0;
        long ans = 1;
        int curs = 0;
        for (int i = 0;i < s.length(); i++) {
            if (s.charAt(i) == 'S') {
                if (cur == 2) {
                    list.add(vv == 0 ? 1 : vv + 1);
                    cur = 0;
                    vv = 0;
                }
                curs++;
                cur++;
            }
            if (cur == 2 && s.charAt(i) == 'P') {
                vv++;
            }
        }
        if (curs % 2 != 0) {
            return 0;
        }
        for (int i = 0;i < list.size(); i++) {
            ans *= list.get(i);
            ans %= mod;
        }
        return (int) ans;
    }

    public int countElements(int[] nums) {
        Map<Integer, Integer> map = new TreeMap<>();
        Set<Integer> set = new TreeSet<>();
        for (int i = 0;i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
            set.add(nums[i]);
        }
        List<Integer> list = new ArrayList<>(set);
        map.remove(list.get(0));
        map.remove(list.get(list.size() - 1));
        int ans = 0;
        for (int x : map.keySet()) {
            ans += map.get(x);
        }
        return  ans;
    }

    public int[] rearrangeArray1(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        for (int i = 0;i < n; i++) {
            if (nums[i] > 0) {
                list1.add(nums[i]);
            } else {
                list2.add(nums[i]);
            }
        }
        int index = 0;
        int v = 0;
        while (index < n) {
            ans[index++] = list1.get(v);
            ans[index++] = list2.get(v);
            v++;
        }
        return ans;
    }

    public List<Integer> findLonely(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        List<Integer> list = new ArrayList<>();
        for (int i = 0;i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        for (int x : map.keySet()) {
            if (!map.containsKey(x - 1) && !map.containsKey(x + 1) && map.get(x) == 1) {
                list.add(x);
            }
        }
        return list;
    }

    public int maximumGood(int[][] statements) {
        int n = statements.length;
        int ans = 0;
        for (int i = 1;i < (1 << n); i++) {
            boolean s = true;
            for (int j = 0;j < n; j++) {
                if (((i >> j) & 1) == 1) {
                    for (int k = 0;k < n; k++) {
                        if (statements[j][k] == 0 && ((i >> k) & 1) == 1) {
                            s = false;
                        }
                        if (statements[j][k] == 1 && ((i >> k) & 1) != 1) {
                            s = false;
                        }
                    }
                }
                if (!s) {
                    break;
                }
            }
            if (s) {
                int cur = 0;
                int vv = i;
                while (vv > 0) {
                    if ((vv & 1) == 1) {
                        cur++;
                    }
                    vv >>= 1;
                }
                ans = Math.max(ans, cur);
            }
        }
        return ans;
    }

    public List<List<Integer>> subsets(int[] nums) {
        int n = nums.length;
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0;i < (1 << n); i++) {
            List<Integer> list = new ArrayList<>();
            for (int j = 0;j < n; j++) {
                if (((i >> j) & 1) == 1) {
                    list.add(nums[j]);
                }
            }
            ans.add(list);
        }
        return ans;
    }

    public int numberOfWeakCharacters(int[][] properties) {
        Arrays.sort(properties, (o1, o2) -> {
            if (o1[0] != o2[0]) {
                return o2[0] - o1[0];
            }
            return o1[1] - o2[1];
        });
        int maxDef = 0;
        int ans = 0;
        int n = properties.length;
        for (int i = 0;i < n; i++) {
            if (properties[i][1] < maxDef) {
                ans++;
            } else {
                maxDef = properties[i][1];
            }
        }
        return ans;
    }

    public List<Integer> maxScoreIndices(int[] nums) {
        int n = nums.length;
        int[] arr1 = new int[n + 1];
        int[] arr2 = new int[n + 1];
        List<Integer> list = new ArrayList<>();
        for (int i = 1;i <= n; i++) {
            arr1[i] = arr1[i - 1] + nums[i - 1];
        }
        int max = Math.max(arr1[n], n - arr1[n]);
        for (int i = 1;i < n; i++) {
            int left = i - arr1[i];
            int right = arr1[n] - arr1[i];
            int vv = left + right;
            max = Math.max(vv, max);
            arr2[i] = vv;
        }
        for (int i = 0;i <= n; i++) {
            if (arr2[i] == max) {
                list.add(i);
            }
        }
        return list;
    }

    public String[] uncommonFromSentences(String s1, String s2) {
        String[] strings1 = s1.split(" ");
        String[] strings2 = s2.split(" ");
        Map<String, Integer> map1 = new HashMap<>();
        Map<String, Integer> map2 = new HashMap<>();
        List<String> list = new ArrayList<>();
        for (int i = 0;i < strings1.length; i++) {
            map1.put(strings1[i], map1.getOrDefault(strings1[i], 0) + 1);
        }
        for (int i = 0;i < strings2.length; i++) {
            map2.put(strings2[i], map2.getOrDefault(strings2[i], 0) + 1);
        }
        for (String x : map1.keySet()) {
            if (map1.get(x) == 1 && !map2.containsKey(x)) {
                list.add(x);
            }
        }
        for (String x : map2.keySet()) {
            if (map2.get(x) == 1 && !map1.containsKey(x)) {
                list.add(x);
            }
        }
        String[] ans = new String[list.size()];
        for (int i = 0;i < list.size(); i++) {
            ans[i] = list.get(i);
        }
        return ans;
    }

    public int numberOfSteps(int num) {
        int ans = 0;
        while (num > 0) {
            if (num % 2 == 0) {
                num >>= 1;
            } else {
                num -= 1;
            }
            ans++;
        }
        return ans;
    }

    public String longestNiceSubstring(String s) {
        int n = s.length();
        int maxPos = 0;
        int maxLen = 0;
        for (int i = 0;i < n; i++) {
            int lower = 0;
            int upper = 0;
            for (int j = i;j < n; j++) {
                if (Character.isLowerCase(s.charAt(j))) {
                    lower |= (1 << (s.charAt(j) - 'a'));
                } else {
                    upper |= (1 << (s.charAt(j) - 'A'));
                }
                if (lower == upper && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    maxPos = i;
                }
            }
        }
        return s.substring(maxPos, maxPos + maxLen);
    }

    public int maxNumberOfBalloons(String text) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0;i < text.length(); i++) {
            map.put(text.charAt(i), map.getOrDefault(text.charAt(i), 0) + 1);
        }
        try {
            int cur = Math.min(map.get('b'), map.get('a'));
            cur = Math.min(cur, map.get('l') >> 1);
            cur = Math.min(cur, map.get('o') >> 1);
            return Math.min(cur, map.get('n'));
        } catch (Exception e) {
            return 0;
        }

    }

    public int countOperations(int num1, int num2) {
        int ans = 0;
        while (num1 != 0 && num2 != 0) {
            if (num1 >= num2) {
                num1 = num1 - num2;
            } else {
                num2 = num2 - num1;
            }
            ans++;
        }
        return ans;
    }

    public long minimumRemoval(int[] beans) {
        Arrays.sort(beans);
        int sum = 0;
        for (int x : beans) {
            sum += x;
        }
        int min = sum;
        int temp = 0;
        for (int i = 0;i < beans.length; i++) {
            temp = sum - (beans.length - i) * beans[i];
            min = Math.min(temp, min);
        }
        return min;
    }

    public List<Integer> luckyNumbers (int[][] matrix) {
        List<Integer> list = new ArrayList<>();
        int[] maxArr = new int[matrix[0].length];
        int[] minArr = new int[matrix.length];

        for (int i = 0;i < matrix.length; i++) {
            int min = Integer.MIN_VALUE;
            for (int j = 0;j < matrix[0].length; j++) {
                min = Math.min(matrix[i][j], min);
            }
            minArr[i] = min;
        }
        for (int j = 0;j < matrix[0].length; j++) {
            int max = Integer.MAX_VALUE;
            for (int i = 0;i < matrix.length; i++) {
                max = Math.max(matrix[j][i], max);
            }
            maxArr[j] = max;
        }
        for (int i = 0;i < matrix.length; i++) {
            for (int j = 0;j < matrix[0].length; j++) {
                if (matrix[i][j] == maxArr[j] && matrix[i][j] == minArr[i]) {
                    list.add(matrix[i][j]);
                }
            }
        }
        return list;
    }
    
    static int[][] dirs = new int[][]{
            {2, 1},{2, -1},{-2, 1},{-2, -1},
            {1, 2},{-1, 2},{1, -2},{-1, -2}
    };

    public double knightProbability(int n, int k, int row, int column) {
        double[][][] dp = new double[k + 1][n][n];
        for (int i = 0;i <= k; i++) {
            for (int j = 0;j < n; j++) {
                for (int z = 0;z < n; z++) {
                    if (i == 0) {
                        dp[i][j][z] = 1;
                    } else {
                        for (int[] dir : dirs) {
                            int nx = i + dir[0];
                            int ny = i + dir[1];
                            if (nx < 0 || nx >= n || ny < 0 || ny >= n) {
                                continue;
                            }
                            dp[i][j][z] = dp[i - 1][nx][ny] / 8.0;
                        }
                    }
                }
            }
        }
        return dp[k][row][column];
    }

    public int countPairs(int[] nums, int k) {
        int ans = 0;
        for (int i = 0;i < nums.length; i++) {
            for (int j = i + 1;j < nums.length; j++) {
                if ((nums[i] + nums[j]) % k == 0) {
                    ans++;
                }
            }
        }
        return ans;
    }

    public long[] sumOfThree(long num) {
        if (num % 3 != 0) {
            return new long[]{};
        }
        long cur = num / 3;
        long[] ans = new long[3];
        ans[0] = cur - 1;
        ans[1] = cur;
        ans[2] = cur + 1;
        return ans;
    }

    public List<Long> maximumEvenSplit(long finalSum) {
        if (finalSum % 2 != 0) {
            return new ArrayList<>();
        }
        long cur = finalSum / 2;
        int vv = 1;
        List<Long> ans = new ArrayList<>();
        while (true) {
            if (vv <= cur) {
                ans.add(vv * 2L);
            } else {
                ans.set(ans.size() - 1, ans.get(ans.size() - 1) + cur * 2L);
                break;
            }
            if (cur == 0) {
                break;
            }
            cur -= vv;
            vv++;
        }
        return ans;
    }

    public long goodTriplets(int[] nums1, int[] nums2) {
        int[] arr = new int[101010];
        BITree biTree = new BITree();
        long[] v = new long[101010];
        long ans = 0;
        int n = nums1.length;
        for (int i = 0;i < n; i++) {
            arr[nums2[i]] = i;
        }
        for (int i = 0;i < n; i++) {
            v[i] = biTree.calc(arr[nums1[i]] + 1);
            biTree.inc(arr[nums1[i]] + 1);
        }
        biTree.clear();
        for (int i = n - 1;i >= 0; i--) {
            v[i] *= biTree.calc(100000) - biTree.calc(arr[nums1[i]] + 1);
            biTree.inc(arr[nums1[i]] + 1);
        }
        for (int i = 0;i < n; i++) {
            ans += v[i];
        }
        return ans;
    }

    public int countEven(int num) {
        int ans = 0;
        for (int i = 1;i <= num; i++) {
                int x = i;
                int cur = 0;
                while (x > 0) {
                     cur += x % 10;
                     x = x / 10;
                }
                if (cur % 2 == 0) {
                    ans++;
                }
        }
        return ans;
    }

    public ListNode mergeNodes(ListNode head) {
        List<Integer> list = new ArrayList<>();
        ListNode hair = new ListNode(0);
        ListNode p = head;
        int cur = 0;
        while (p != null) {
            if (p.val != 0) {
                cur += p.val;
            } else {
                if (cur != 0) {
                    list.add(cur);
                    cur = 0;
                }
            }
            p = p.next;
        }
        p = hair;
        for (int i = 0;i < list.size(); i++) {
            p.next = new ListNode(list.get(i));
            p = p.next;
        }
        return hair.next;
    }

    public String repeatLimitedString(String s, int repeatLimit) {
        PriorityQueue<PP> que = new PriorityQueue<>((o1, o2) -> o2.x - o1.x);
        int[] arr = new int[26];
        for (int i = 0;i < s.length(); i++) {
            arr[s.charAt(i) - 'a']++;
        }
        for (int i = 0;i < 26; i++) {
            if (arr[i] != 0)
                que.add(new PP((char) ('a' + i), arr[i]));
        }
        StringBuilder sb = new StringBuilder();
        while (true) {
            PP p1 = que.poll();
            if (p1.y <= repeatLimit) {
                for (int i = 0;i < p1.y; i++) {
                    sb.append(p1.x);
                }
            } else {
                for (int i = 0;i < repeatLimit; i++) {
                    sb.append(p1.x);
                }
                if (que.size() >= 1) {
                    PP p2 = que.poll();
                    sb.append(p2.x);
                    que.add(new PP(p1.x, p1.y - repeatLimit));
                    if (p2.y - 1 > 0) {
                        que.add(new PP(p2.x, p2.y - 1));
                    }
                } else {
                    break;
                }
            }
            if (que.isEmpty()) {
                break;
            }
        }
        return sb.toString();
    }

    class PP {
        char x;
        int y;

        public PP(char x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    public long coutPairs(int[] nums, int k) {
        int n = nums.length;
        int[] arr = new int[101010];
        int[] f = new int[101010];
        for (int i = 0;i < n; i++) {
            ++arr[nums[i]];
        }
        for (int i = 1;i <= 100000; i++) {
            for (int j = i;j <= 100000; j+=i) {
                f[i] += arr[j];
            }
        }
        long ans = 0;
        for (int i = 0;i < n; i++) {
            ans += f[k / (gcd(k, nums[i]))];
            if (((long) nums[i] * nums[i]) % k == 0) {
                ans--;
            }
        }
        return ans / 2;
    }

    public String reverseOnlyLetters(String s) {
        List<Character> list = new ArrayList<>();
        for (int i = 0;i < s.length(); i++) {
            char x = s.charAt(i);
            if ((x >= 'a' && x <= 'z') || (x >= 'A' && x <= 'Z')) {
                list.add(x);
            }
        }
        StringBuilder sb = new StringBuilder();
        int v = list.size();
        for (int i = 0;i < s.length(); i++) {
            char x = s.charAt(i);
            if ((x >= 'a' && x <= 'z') || (x >= 'A' && x <= 'Z')) {
                sb.append(list.get(--v));
            } else {
                sb.append(x);
            }
        }
        return sb.toString();
    }

    public int prefixCount(String[] words, String pref) {
        int ans = 0;
        for (int i = 0;i < words.length; i++) {
            if (words[i].startsWith(pref)) {
                ans++;
            }
        }
        return ans;
    }

    public int minSteps(String s, String t) {
        int[] arr1 = new int[26];
        int[] arr2 = new int[26];
        for (int i = 0;i < s.length(); i++) {
            arr1[s.charAt(i) - 'a']++;
        }
        for (int i = 0;i < t.length(); i++) {
            arr2[t.charAt(i) - 'a']++;
        }
        int ans = 0;
        for (int i = 0;i < 26; i++) {
            ans += Math.abs(arr1[i] - arr2[i]);
        }
        return ans;
    }

    public long minimumTime(int[] time, int totalTrips) {
        long left = 0, right = (long) 1e14 + 10;
        while (left < right) {
            long mid = (left + right) >> 1;
            if (check(mid, totalTrips, time)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    public boolean check(long mid, int total, int[] time) {
        long ans = 0;
        for (int i = 0;i < time.length; i++) {
            ans += time[i] / mid;
        }
        return ans >= total;
    }

    void dfs(ArrayList<ArrayList<Integer>> grid, int r, int c) {
        int nr = grid.size();
        int nc = grid.get(0).size();

        if (r < 0 || c < 0 || r >= nr || c >= nc || grid.get(r).get(c) == '0') {
            return;
        }

        grid.get(r).set(c, 0);
        dfs(grid, r - 1, c);
        dfs(grid, r + 1, c);
        dfs(grid, r, c - 1);
        dfs(grid, r, c + 1);
    }

    public int numIslands(ArrayList<ArrayList<Integer>> grid) {
        if (grid == null || grid.size() == 0) {
            return 0;
        }

        int nr = grid.size();
        int nc = grid.get(0).size();
        int num_islands = 0;
        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < nc; ++c) {
                if (grid.get(r).get(c) == 1) {
                    ++num_islands;
                    dfs(grid, r, c);
                }
            }
        }

        return num_islands;
    }

    public int maximumRequests(int n, int[][] requests) {
        int m = requests.length;
        int ans = 0;
        for (int i = 0;i < (1 << m); i++) {
            int cnt = Integer.bitCount(i);
            if (cnt < ans)
                continue;
            if (check(i, requests)) {
                ans = cnt;
            }
        }
        return ans;
    }

    private boolean check(int cur, int[][] requests) {
        int[] res = new int[20];
        int sum = 0;
        for (int i = 0;i < 16; i++) {
            if (((cur >> i) & 1) == 1) {
                if (++res[requests[i][0]] == 1) {
                    sum++;
                }
                if (--res[requests[i][1]] == 0) {
                    sum--;
                }
            }
        }
        return sum == 0;
    }

    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new TreeSet<>();
        for (int i = 0;i < nums.length; i++) {
            set.add(nums[i]);
        }
        int flag = 0;
        int last = -1;
        int cur = 0;
        int ans = 0;
        for (int x : set) {
            if (flag == 0) {
                flag = 1;
                cur = 1;
                last = x;
            } else {
                if (x - last == 1) {
                    cur++;
                    last = x;
                } else {
                    ans = Math.max(cur, ans);
                    cur = 1;
                    last = x;
                }
            }
        }
        ans = Math.max(cur, ans);
        return ans;
    }

    public String convert(String s, int r) {
        if (r < 2) {
            return s;
        }
        List<StringBuilder> list = new ArrayList<>();
        for (int i = 0;i < r; i++) {
            list.add(new StringBuilder());
        }
        for (int i = 0, x = 0, t = r * 2 - 2;i < s.length(); i++) {
            list.get(x).append(s.charAt(i));
            if (i % t < r - 1) {
                x++;
            } else {
                x--;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (StringBuilder x : list) {
            sb.append(x);
        }
        return sb.toString();
    }

    private int countSum = 0;

    public int sumNumbers(TreeNode root) {
        dfsTree(root, 0);
        return countSum;
    }

    public void dfsTree(TreeNode root, int cur) {
        if (root.left == null && root.right == null) {
            cur = cur * 10 + root.val;
            countSum += cur;
            return ;
        }
        cur = cur * 10 + root.val;
        if (root.left != null) {
            dfsTree(root.left, cur);
        }
        if (root.right != null) {
            dfsTree(root.right, cur);
        }
    }

    public int mostFrequent(int[] nums, int key) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0;i < nums.length; i++) {
            if (i + 1 < nums.length) {
                if (nums[i] == key) {
                    map.put(nums[i + 1], map.getOrDefault(nums[i+1], 0) + 1);
                }
            }
        }
        int ans = 0;
        int max = -1;
        for (Integer x : map.keySet()) {
            if (map.get(x) > max) {
                ans = x;
                max = map.get(x);
            }
        }
        return ans;
    }

    public int[] sortJumbled(int[] mapping, int[] nums) {
        VV[] vvs = new VV[nums.length];
        for (int i = 0;i < nums.length; i++) {
            int s = getNewNumser(mapping, nums[i]);
            vvs[i] = new VV(nums[i], i, s);
        }
        Arrays.sort(vvs, new Comparator<VV>() {
            @Override
            public int compare(VV o1, VV o2) {
                if (o1.z != o2.z) {
                    return o1.z - o2.z;
                }
                return o1.y - o2.y;
            }
        });
        int[] ans = new int[nums.length];
        for (int i = 0;i < nums.length; i++) {
            ans[i] = vvs[i].x;
        }
        return ans;
    }

    private int getNewNumser(int[] mapping, int num) {
        try {
            StringBuilder sb = new StringBuilder();
            while (num >= 0) {
                sb.append(mapping[num % 10]);
                num /= 10;
            }
            sb.reverse();
            return Integer.parseInt(sb.toString());
        } catch (Exception e) {
            return 0;
        }

    }

    class VV {
        int x, y, z;

        public VV(int x, int y, int z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }
    }

    public List<List<Integer>> getAncestors(int n, int[][] edges) {
        List<List<Integer>> list = new ArrayList<>();
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0;i < edges.length; i++) {
            int x = edges[i][0];
            int y = edges[i][1];
            map.putIfAbsent(y, new ArrayList<>());
            map.get(y).add(x);
        }
        Set<Integer> set = new TreeSet<>();
        Queue<Integer> que = new ArrayDeque<>();
        for (int i = 0;i < n; i++) {
            que.add(i);
            list.add(new ArrayList<>());
            while (!que.isEmpty()) {
                int vv = que.poll();
                if (map.get(vv) == null) {
                    continue;
                }
                for (int x : map.get(vv)) {
                    set.add(x);
                    que.add(x);
                }
            }
            for (int x : set) {
                list.get(i).add(x);
            }
            set.clear();
            que.clear();
        }
        return list;
    }

    public int minMovesToMakePalindrome(String s) {
        int n = s.length();
        int e = n - 1;
        int res = 0;
        int odd = 0;
        StringBuilder sb = new StringBuilder(s);
        for (int i = 0;i < e; i++) {
            for (int j = e;j >= i; j--) {
                if (i == j) {
                    if (n % 2 == 0 || odd == 1) {
                        return 0;
                    }
                    odd = 1;
                    res += n / 2 - i;
                } else if (sb.charAt(i) == sb.charAt(j)) {
                    for (int k = j;k < e; k++) {
                        char t = sb.charAt(k);
                        sb.setCharAt(k, sb.charAt(k + 1));
                        sb.setCharAt(k + 1, t);
                        res++;
                    }
                    e--;
                    break;
                }
            }
        }
        return res;
    }

    public List<String> cellsInRange(String s) {
        List<String> ans = new ArrayList<>();
        String[] strings = s.split(":");
        int x1 = strings[0].charAt(0) - 'A';
        int  y1 = strings[0].charAt(1) - '0';
        int x2 = strings[1].charAt(0) - 'A';
        int  y2 = strings[1].charAt(1) - '0';
        for (int i = x1;i <= x2; i++) {
            for (int j = y1;j <= y2; j++) {
                StringBuilder sb = new StringBuilder();
                sb.append((char)('A' + i)).append(j);
                ans.add(sb.toString());
            }
        }
        return ans;
    }

    public long minimalKSum(int[] nums, int k) {
        Set<Integer> set = new TreeSet<>();
        for (int i = 0;i < nums.length; i++) {
            set.add(nums[i]);
        }
        int last = -1;
        long ans = 0;
        for (int x : set) {
            if (last == -1) {
                if (x != 1) {
                    int e = x - 1;
                    if (e > k) {
                        e = 1 + k - 1;
                        k = 0;
                    } else {
                        k -= e;
                    }
                    ans += getNumss(1, e);
                }
            } else {
                int e = x - 1;
                int s = last + 1;
                if (e - s + 1 > k) {
                    e = s + k - 1;
                    k = 0;
                } else {
                    k -= e - s + 1;
                }
                ans += getNumss(s, e);
            }
            last = x;
            if (k == 0) {
                break;
            }
        }
        if (k != 0) {
            ans += getNumss(last + 1, last + k);
        }
        return ans;
    }

    public long getNumss(int s, int e) {
        return (long) (e - s + 1) * (s + e) / 2;
    }

    public TreeNode createBinaryTree(int[][] descriptions) {
        Map<Integer, List<Pair>> map = new HashMap<>();
        Set<Integer> set = new HashSet<>();
        for (int[] des : descriptions) {
            int u = des[0];
            int v = des[1];
            int w = des[2];
            map.putIfAbsent(u, new ArrayList<>());
            map.get(u).add(new Pair(v, w));
            set.add(v);
        }
        int rootVal = 0;
        for (int[] des : descriptions) {
            int u = des[0];
            if (!set.contains(u)) {
                rootVal = u;
                break;
            }
        }
        TreeNode root = new TreeNode(rootVal);
        dfs(map, root, rootVal);
        return root;
    }

    private void dfs(Map<Integer, List<Pair>> map, TreeNode root, int val) {
        if (map.get(val) == null) {
            return ;
        }
        for (Pair pair : map.get(val)) {
            int v = pair.x;
            int w = pair.y;
            if (w == 1) {
                root.left = new TreeNode(v);
                dfs(map, root.left, v);
            } else {
                root.right = new TreeNode(v);
                dfs(map, root.right, v);
            }
        }
    }

    public List<Integer> replaceNonCoprimes(int[] nums) {
        List<Integer> list = new LinkedList<>();
        for (int i = 0;i < nums.length; i++) {
            list.add(nums[i]);
        }
        int index = 0;
        while (true) {
            if (index + 1 < list.size()) {
                int x = list.get(index);
                int y = list.get(index + 1);
                int g = gcd(x, y);
                if (g == 1) {
                    index++;
                } else {
                    int ne = (x / g) * y;
                    list.remove(index + 1);
                    list.set(index, ne);
                }
            } else {
                break;
            }
        }
        return list;
    }

    public int[] platesBetweenCandles(String s, int[][] queries) {
        int n = s.length();
        int m = queries.length;
        int[] sum = new int[n + 1];
        int[] ans = new int[m];
        List<Integer> list = new ArrayList<>();
        for (int i = 0;i < n; i++) {
            if (s.charAt(i) == '|') {
                list.add(i);
            }
            sum[i + 1] = sum[i] + (s.charAt(i) == '*' ? 1 : 0);
        }
        if (list.size() == 0) {
            return ans;
        }
        for (int i = 0;i < m; i++) {
            int a = queries[i][0], b = queries[i][1];
            int c = -1, d = -1;
            int l = 0, r = list.size();
            while (l < r) {
                int mid = (l + r) >> 1;
                if(list.get(mid) >= a) r = mid;
                else l = mid + 1;
            }
            if (list.get(l) >= a) c = list.get(l);
            else continue;
            l = 0;
            r = list.size();
            while (l < r) {
                int mid = (l + r) >> 1;
                if (list.get(mid) <= b) l = mid;
                else r = mid - 1;
            }
            if (list.get(r) <= b) d = list.get(r);
            else continue;
            if (c <= d) {
                ans[i] = sum[d + 1] - sum[c];
            }
        }
        return ans;
    }

    private List<Integer> list = new ArrayList<>();

    public List<Integer> postorder(Node root) {
        if (root == null) {
            return list;
        }
        pos(root);
        return list;
    }

    private void pos(Node root) {
        if (root == null) {
            return ;
        }
        for (int i = 0;i < root.children.size(); i++) {
            pos(root.children.get(i));
        }
        list.add(root.val);
    }

    public List<Integer> findKDistantIndices(int[] nums, int key, int k) {
        int n = nums.length;
        List<Integer> list = new ArrayList<>();
        for (int i = 0;i < n; i++) {
            int cur = Math.max(i - k, 0);
            int cur2 = Math.min(i + k, n);
            for (int j = cur;j <= cur2; j++) {
                if (nums[j] == key) {
                    list.add(i);
                    break;
                }
            }
        }
        return list;
    }

    public int digArtifacts(int n, int[][] artifacts, int[][] dig) {
        int[][] arr = new int[n][n];
        for (int i = 0;i < dig.length; i++) {
            int x = dig[i][0];
            int y = dig[i][1];
            arr[x][y] = 1;
        }
        int ans = 0;
        for (int i = 0;i < artifacts.length; i++) {
            int x1 = artifacts[i][0];
            int y1 = artifacts[i][1];
            int x2 = artifacts[i][2];
            int y2 = artifacts[i][3];
            boolean flag = true;
            for (int j = x1;j <= x2; j++) {
                for (int k = y1;k <= y2; k++) {
                    if (arr[j][k] == 0) {
                        flag = false;
                        break;
                    }
                }
            }
            if (flag) {
                ans++;
            }
        }
        return ans;
    }

    public int maximumTop(int[] nums, int k) {
        if (k == 0) {
            return nums[0];
        }
        if (nums.length == 1 && k == 1) {
            return -1;
        }
        if (nums.length == 1 && k % 2 == 1) {
            return -1;
        }
        if (k < nums.length) {
            int ans = -1;
            for (int i = 0;i < k - 1; i++) {
                ans = Math.max(nums[i], ans);
            }
            return Math.max(ans, nums[k]);
        } else if (k == nums.length){
            int ans = -1;
            for (int i = 0;i < nums.length - 1; i++) {
                ans = Math.max(nums[i], ans);
            }
            return ans;
        } else {
            int ans = -1;
            for (int i = 0;i < nums.length; i++) {
                if (nums[i] > ans) {
                    ans = nums[i];
                }
            }
            return ans;
        }
    }

    public long minimumWeight(int n, int[][] edges, int src1, int src2, int dest) {
        Map<Integer, List<Pair>> map = new HashMap<>();
        Map<Integer, List<Pair>> map1 = new HashMap<>();
        for (int[] edge : edges) {
            int x = edge[0];
            int y = edge[1];
            int w = edge[2];
            map.putIfAbsent(x, new ArrayList<>());
            map.get(x).add(new Pair(y, w));
            map1.putIfAbsent(y, new ArrayList<>());
            map1.get(y).add(new Pair(x, w));
        }
        long[] dis1 = dijkstra(src1, map);
        long[] dis2 = dijkstra(src2, map);
        long[] dis3 = dijkstra(dest, map1);
        long ans = (long) 1e15;
        for (int i = 0;i < n; i++) {
            ans = Math.min(ans, (dis1[i] + dis2[i] + dis3[i]));
        }
        return ans >= (long) 1e11 ? -1 : ans;
    }

    public long[] dijkstra(int s, Map<Integer, List<Pair>> map) {
        long[] dis = new long[101010];
        int[] vis = new int[101010];
        Arrays.fill(dis, (long) 1e15);
        dis[s] = 0;
        Queue<HeapNode> que = new PriorityQueue<>(Comparator.comparingLong(o -> o.val));
        que.add(new HeapNode(s, 0));
        while (!que.isEmpty()) {
            HeapNode node = que.poll();
            if (vis[node.u] == 1) {
                continue;
            }
            vis[node.u] = 1;
            if (map.get(node.u) == null) {
                continue;
            }
            for (Pair pair : map.get(node.u)) {
                int v = pair.x;
                if (dis[v] > dis[node.u] + pair.y) {
                    dis[v] = dis[node.u] + pair.y;
                    que.add(new HeapNode(v, dis[v]));
                }
            }
        }
        return dis;
    }

    public boolean divideArray(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int x : nums) {
            map.put(x, map.getOrDefault(x, 0) + 1);
        }
        for (int x : map.keySet()) {
            if (map.get(x) % 2 != 0) {
                return false;
            }
        }
        return true;
    }

    public long maximumSubsequenceCount(String text, String pattern) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0;i < text.length(); i++) {
            char x = text.charAt(i);
            if (x == pattern.charAt(0) || x == pattern.charAt(1)) {
                sb.append(x);
            }
        }
        if (pattern.charAt(0) == pattern.charAt(1)) {
            long cur = sb.length();
            return cur * (cur + 1) / 2;
        }
        StringBuilder sb1 = new StringBuilder(pattern.charAt(0));
        sb1.append(sb);
        StringBuilder sb2 = new StringBuilder(sb);
        sb2.append(pattern.charAt(1));
        long cur1 = 0;
        long vv1 = 0;
        long cur2 = 0;
        long vv2 = 0;
        for (int i = 0;i < sb1.length(); i++) {
            if (sb1.charAt(i) == pattern.charAt(1)) {
                cur1++;
            }
            if (sb2.charAt(i) == pattern.charAt(1)) {
                cur2++;
            }
        }
        for (int i = 0;i < sb1.length(); i++) {
            if (sb1.charAt(i) == pattern.charAt(1)) {
                --cur1;
            } else {
                vv1 += cur1;
            }
            if (sb2.charAt(i) == pattern.charAt(1)) {
                --cur2;
            } else {
                vv2 += cur2;
            }
        }
        return Math.max(cur1, cur2);
    }

    public int halveArray(int[] nums) {
        Queue<Double> que = new PriorityQueue<>((o1, o2) -> {
            if (o2 > o1) {
                return 1;
            }
            return -1;
        });
        double cur = 0;
        for (int i = 0;i < nums.length; i++) {
            cur += nums[i] * 1.0;
            que.add((double) nums[i]);
        }
        double vv = cur;
        int ans = 0;
        while (vv > cur / 2) {
            double x = que.poll();
            que.add(x / 2);
            vv -= (x / 2);
            ans++;
        }
        return ans;
    }

    public int countHillValley(int[] nums) {
        int ans = 0;
        List<Integer> list = new ArrayList<>();
        for (int i = 0;i < nums.length - 1; i++) {
            if (nums[i] == nums[i + 1]) {
                int v = 2;
                while (i + v < nums.length && nums[i + v] == nums[i]) {
                    v++;
                }
                i += v - 1;
            }
            list.add(nums[i]);
        }
        if (nums[nums.length - 1] != nums[nums.length - 2]) {
            list.add(nums[nums.length - 1]);
        }
        for (int i = 1;i < list.size() - 1; i++) {
            if (list.get(i) < list.get(i - 1) && list.get(i) < list.get(i + 1)) {
                ans++;
            }
            if (list.get(i) > list.get(i - 1) && list.get(i) > list.get(i + 1)) {
                ans++;
            }
        }
        return ans;
    }

    public int countCollisions(String directions) {
        int ans = 0;
        boolean flag = false;
        int l = 0;
        int r = 0;
        for (int i = 0;i < directions.length(); i++) {
            char cur = directions.charAt(i);
            if (cur == 'L' && flag) {
                if (r == 0) {
                    ans++;
                } else {
                    ans += 2;
                    ans += r - 1;
                }
            }
            if (cur == 'R') {
                flag = true;
                r++;
            }
            if (cur == 'S' && r != 0) {
                ans += r;
                r = 0;
            }
        }
        return ans;
    }

    public int[] maximumBobPoints(int numArrows, int[] aliceArrows) {
        int[] ans = new int[11];
        int max = 0;
        for (int i = 4091;i < (1 << 12); i++) {
            int cur = 0;
            int curArrows = numArrows;
            for (int j = 0;j < 12; j++) {
                if (((i >> j) & 1) == 1) {
                    if (curArrows >= aliceArrows[j] + 1) {
                        cur += j;
                        curArrows -= aliceArrows[j] + 1;
                    }
                }
            }
            if (cur > max) {
                max = cur;
                curArrows = numArrows;
                Arrays.fill(ans, 0);
                for (int j = 0;j < 12; j++) {
                    if (((i >> j) & 1) == 1) {
                        if (curArrows >= aliceArrows[j] + 1) {
                            ans[j] = aliceArrows[j] + 1;
                            curArrows -= aliceArrows[j] + 1;
                        }
                    }
                }
            }
        }
        return ans;
    }

    public List<List<Integer>> findDifference(int[] nums1, int[] nums2) {
        Set<Integer> set1 = new HashSet<>();
        Set<Integer> set2 = new HashSet<>();
        Set<Integer> set3 = new HashSet<>();
        Set<Integer> set4 = new HashSet<>();
        for (int x : nums1) {
            set1.add(x);
        }
        for (int x : nums2) {
            set2.add(x);
        }
        List<List<Integer>> ans = new ArrayList<>();
        for (int x : nums1) {
            if (!set2.contains(x)) {
                set3.add(x);
            }
        }
        for (int x : nums2) {
            if (!set1.contains(x)) {
                set4.add(x);
            }
        }
        ans.add(new ArrayList<>(set3));
        ans.add(new ArrayList<>(set4));
        return ans;
    }

    public int minDeletion(int[] nums) {
        int index = 0;
        int ans = 0;
        for (int i = 0;i < nums.length - 1; i++) {
            if (index % 2 == 0) {
                if (nums[i] == nums[i + 1]) {
                    ans++;
                    continue;
                }
            }
            index++;
        }
        if (index % 2 == 0) {
            ans++;
        }
        return ans;
    }

    public long[] kthPalindrome(int[] queries, int intLength) {
        long[] ans = new long[queries.length];
        int x = (intLength + 1) / 2;
        long max = (long) (9 * Math.pow(10, x - 1));
        long cur = (long) Math.pow(10, x - 1);
        if (intLength == 1) {
            cur = 1;
        }
        for (int i = 0;i < queries.length; i++) {
            if (queries[i] > max) {
                ans[i] = -1;
            } else {
                long vv = cur + queries[i] - 1;
                long cura = vv;
                if (intLength % 2 != 0) {
                    cura /= 10;
                }
                StringBuilder sb = new StringBuilder(String.valueOf(vv));
                if (intLength != 1) {
                    StringBuilder sb1 = new StringBuilder(String.valueOf(cura));
                    sb.append(sb1.reverse());
                }
                ans[i] = Long.parseLong(sb.toString());
            }
        }
        return ans;
    }

    public int maxConsecutiveAnswers(String answerKey, int k) {
        char[] chars = answerKey.toCharArray();
        int t = 0, f = 0;
        int l = 0, r = 0;
        int ans = 0;
        while (r < chars.length) {
            if (chars[r++] == 'T') {
                t++;
            } else {
                f++;
            }
            while (t < k && f < k) {
                if (chars[l++] == 'T') {
                    t--;
                } else {
                    f--;
                }
            }
            ans = Math.max(ans, r - l + 1);
        }
        return ans;
    }

    public List<Integer> busiestServers(int k, int[] arrival, int[] load) {
        TreeSet<Integer> set = new TreeSet<>();
        for (int i = 0;i < k; i++) {
            set.add(i);
        }
        Queue<int[]> que = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        int[] requests = new int[k];
        for (int i = 0;i < arrival.length; i++) {
            while (!que.isEmpty() && que.peek()[0] < arrival[i]) {
                set.add(que.poll()[1]);
            }
            if (set.isEmpty()) {
                continue;
            }
            Integer cur = set.ceiling(i % k);
            if (cur == null) {
                cur = set.first();
            }
            requests[cur]++;
            que.add(new int[] {arrival[i] + load[i], cur});
            set.remove(cur);
        }
        int max = Arrays.stream(requests).max().getAsInt();
        List<Integer> ans = new ArrayList<>();
        for (int i = 0;i < k; i++) {
            if (requests[i] == max) {
                ans.add(i);
            }
        }
        return ans;
    }

    public int minBitFlips(int start, int goal) {
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        while (start > 0) {
            sb1.append(start % 2);
            start /= 2;
        }
        while (goal > 0) {
            sb2.append(goal % 2);
            goal /= 2;
        }
        int max = Math.max(sb1.length(), sb2.length());
        for (int i = sb1.length();i < max; i++) {
            sb1.append('0');
        }
        for (int i = sb2.length();i < max; i++) {
            sb2.append('0');
        }
        int ans = 0;
        for (int i = 0;i < max; i++) {
            if (sb1.charAt(i) != sb2.charAt(i)) {
                ans++;
            }
        }
        return ans;
    }

    public int triangularSum(int[] nums) {
        int n = nums.length;
        int[][] arr = new int[n][n];
        int cur = n - 1;
        for (int i = 0;i < n; i++) {
            arr[0][i] = nums[i];
        }
        for (int i = 1;i < n; i++) {
            for (int j = 0;j <= cur; j++) {
                arr[i][j] = (arr[i - 1][j] + arr[i - 1][j + 1])%10;
            }
            cur--;
        }
        return arr[n - 1][0];
    }

    public long numberOfWays1(String s) {
        List<Long> list = new ArrayList<>();
        char last = 0;
        long cur = 1;
        long ans = 0;
        for (int i = 0;i < s.length(); i++) {
            if (i == 0) {
                last = s.charAt(i);
            } else {
                if (s.charAt(i) == last) {
                    cur++;
                } else {
                    list.add(cur);
                    cur = 1;
                    last = s.charAt(i);
                }
            }
        }
        list.add(cur);
        long cur3 = 0;
        long cur4 = 0;
        for (int i = 0;i < list.size(); i++) {
            if (i % 2 != 0) {
                cur3 += list.get(i);
            } else {
                cur4 += list.get(i);
            }
        }
        int index = 0;
        long cur1 = list.get(0);
        cur4 -= cur1;
        long cur2 = 0;
        for (int i = 1;i < list.size(); i++) {
            if (i % 2 == 1) {
                ans += list.get(i) * cur4 * cur1;
                cur3 -= list.get(i);
                cur1 += list.get(i);
            } else {
                ans += list.get(i) * cur3 * cur2;
                cur4 -= list.get(i);
                cur2 += list.get(i);
            }
        }
        return ans;
    }

    public long sumScores(String s) {
        int[] cur = getNext(s);
        long ans = 0;
        for (int i = 0;i < s.length(); i++) {
            ans += cur[i];
        }
        return ans;
    }
    private int[] getNext(String s) {
        int[] next = new int[s.length()+1]; // 表示子串s[0],s[1],...,s[i-1]的最长真前后缀长度
        int i = 0, j = -1; // j和next[0]初始化为-1，目的是防止下标越界
        next[0] = -1;

        while (i < s.length()) {
            if (j == -1 || s.charAt(i) == s.charAt(j)) {
                next[++i] = ++j; // 匹配成功
            }else {
                j = next[j];  // 表示当前前缀字符串匹配失败，回到上一次匹配位置
            }
        }

        return next;
    }

    public List<List<Integer>> findWinners(int[][] matches) {
        List<List<Integer>> ans = new ArrayList<>();
        Map<Integer, Integer> map = new TreeMap<>();
        for (int[] cur : matches) {
            map.putIfAbsent(cur[0], map.getOrDefault(cur[0], 0));
            map.putIfAbsent(cur[1], map.getOrDefault(cur[1], 0) + 1);
        }
        ans.add(new ArrayList<>());
        ans.add(new ArrayList<>());
        for (int x : map.keySet()) {
            if (map.get(x) == 0) {
                ans.get(0).add(x);
            } else if (map.get(x) == 1) {
                ans.get(1).add(x);
            }
        }
        return ans;
    }

    public int maximumCandies(int[] candies, long k) {
        Arrays.sort(candies);
        long l = 0, r = (long) 1e12;
        while (l < r) {
            long mid = (l + r) >> 1;
            if (check1(mid, candies, k)) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return (int) l;
    }

    private boolean check1(long mid, int[] candies, long k) {
        long cur = 0;
        for (int i = 0;i < candies.length; i++) {
            cur += candies[i] / mid;
        }
        return cur >= k;
    }

    class Encrypter {

        Map<Character, String> map1;
        Map<String, Integer> map2;
        Map<String, List<Character>> map3;
        Set<String> set;

        public Encrypter(char[] keys, String[] values, String[] dictionary) {
            set = new HashSet<>();
            map1 = new HashMap<>();
            map2 = new HashMap<>();
            map3 = new HashMap<>();
            for (int i = 0;i < keys.length; i++) {
                map1.put(keys[i], values[i]);
                map2.put(values[i], map2.getOrDefault(values[i], 0) + 1);
                map3.putIfAbsent(values[i], new ArrayList<>());
                map3.get(values[i]).add(keys[i]);
            }
            set.addAll(Arrays.asList(dictionary));
        }

        public String encrypt(String word1) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0;i < word1.length(); i++) {
                sb.append(map1.get(word1.charAt(i)));
            }
            return sb.toString();
        }

        public int decrypt(String word2) {

            int ans = 0;
            for (String s : set) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0;i < s.length(); i++) {
                    sb.append(map1.get(s.charAt(i)));
                }
                if (sb.toString().equals(word2)) {
                    ans++;
                }
            }
            return ans;
        }
    }

    class NumArray {

        private int[] tree;
        private int[] nums;
        private int n;
        private int lowbit(int x) {
            return x & -x;
        }

        private void add(int x, int u) {
            for (int i = x;i <= n; i += lowbit(i)) {
                tree[i] +=  u;
            }
        }

        private int query(int x) {
            int cur = 0;
            for (int i = x;x > 0; i -= lowbit(i)) {
                cur += tree[i];
            }
            return cur;
        }

        public NumArray(int[] nums) {
            this.nums = nums;
            this.n = nums.length;
            for (int i = 0;i < n; i++) {
                add(i + 1, nums[i]);
            }
        }

        public void update(int index, int val) {
            add(index + 1, val - nums[index]);
            nums[index] = val;
        }

        public int sumRange(int left, int right) {
            return query(right + 1) - query(left);
        }
    }

    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] edge : edges) {
            map.putIfAbsent(edge[0], new ArrayList<>());
            map.putIfAbsent(edge[1], new ArrayList<>());
            map.get(edge[0]).add(edge[1]);
            map.get(edge[1]).add(edge[0]);
        }
        int[] par = new int[n];
        int x = findLongestNode(0, map, n, par);
        int y = findLongestNode(x, map, n, par);
        List<Integer> list = new ArrayList<>();
        while (x != y) {
            list.add(y);
            y = par[y];
        }
        list.add(x);
        int len = list.size();
        if (len % 2 == 1) {
            return Collections.singletonList(list.get((len + 1) / 2));
        } else {
            return Arrays.asList(list.get(len / 2), list.get(len / 2 + 1));
        }
    }

    public int findLongestNode(int node, Map<Integer, List<Integer>> map, int n, int[] par) {
        Queue<Integer> que = new LinkedList<>();
        que.add(node);
        int[] vis = new int[n];
        int ans = 0;
        while (!que.isEmpty()) {
            int cur = que.poll();
            ans = cur;
            if (map.get(cur) == null) {
                continue;
            }
            vis[cur] = 1;
            for (int x : map.get(cur)) {
                if (vis[x] != 1) {
                    par[x] = cur;
                    que.add(x);
                }
            }
        }
        return ans;
    }

    public boolean rotateString(String s, String goal) {
        return s.length() == goal.length() && (s + s).contains(goal);
    }

    public List<List<Integer>> levelOrder(Node root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> list = new ArrayList<>();
        Queue<Node> que = new LinkedList<>();
        que.add(root);
        list.add(new ArrayList<>());
        list.get(0).add(root.val);
        int index = 1;
        int last = 1;
        int cur = 0;
        while (!que.isEmpty()) {
            list.add(new ArrayList<>());
            for (int i = 0;i < last; i++) {
                Node node = que.poll();
                if (node.children == null) {
                    continue;
                }
                for (Node x : node.children) {
                    que.add(x);
                    cur++;
                    list.get(index).add(x.val);
                }
            }
            last = cur;
            cur = 0;
            index++;
        }
        return list;
    }

    public int largestInteger(int num) {
        int[] cur = new int[11];
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        List<Integer> list3 = new ArrayList<>();
        List<Integer> list4 = new ArrayList<>();
        int index = 0;
        int vv = num;
        while (vv > 0) {
            if ((vv % 10) % 2 == 0) {
                list1.add(vv % 10);
                list2.add(index);
            } else {
                list3.add(vv % 10);
                list4.add(index);
            }
            vv /= 10;
            index++;
        }
        Collections.sort(list1);
        Collections.sort(list3);
        for (int i = 0;i < list1.size(); i++) {
            cur[list2.get(i)] = list1.get(i);
        }
        for (int i = 0;i < list3.size(); i++) {
            cur[list4.get(i)] = list3.get(i);
        }
        int ans = 0;
        for (int i = index - 1;i >= 0; i++) {
            ans = ans * 10 + cur[i];
        }
        return ans;
    }

    public String minimizeResult(String expression) {
        String[] strings = expression.split("\\+");
        int cur = Integer.parseInt(strings[0]) + Integer.parseInt(strings[1]);
        String ans = "(" + expression + ")";
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder(strings[0]);
        for (int i = 0;i < strings[0].length(); i++) {
            if (i != 0) {
                sb1.append(strings[0].charAt(i - 1));
                sb2.deleteCharAt(0);
            }
            StringBuilder sb3 = new StringBuilder();
            StringBuilder sb4 = new StringBuilder(strings[1]);
            for (int j = 0;j < strings[1].length(); j++) {
                sb3.append(strings[1].charAt(j));
                sb4.deleteCharAt(0);
                int cur1 = 1;
                int cur2 = Integer.parseInt(sb2.toString());
                int cur3 = Integer.parseInt(sb3.toString());
                int cur4 = 1;
                if (sb1.length() != 0) {
                    cur1 = Integer.parseInt(sb1.toString());
                }
                if (sb4.length() != 0) {
                    cur4 = Integer.parseInt(sb4.toString());
                }
                int vv1 = cur1 * (cur2 + cur3) * cur4;
                if (vv1 < cur) {
                    StringBuilder sb = new StringBuilder();
                    if (i == 0) {
                        sb.append("(");
                    }
                    if (sb1.length() != 0) {
                        sb.append(sb1);
                    }
                    if (i != 0) {
                        sb.append("(");
                    }
                    sb.append(sb2).append("+").append(sb3);
                    if (j != strings[1].length()-1) {
                        sb.append(")");
                    }
                    if (sb4.length() != 0) {
                        sb.append(sb4);
                    }
                    if (j == strings[1].length()-1) {
                        sb.append(")");
                    }
                    ans = sb.toString();
                    cur = vv1;
                }
            }
        }
        return ans;
    }

    public int maximumProduct(int[] nums, int k) {
        Queue<Integer> que = new PriorityQueue<>();
        for (int i = 0;i < nums.length; i++) {
            que.add(nums[i]);
        }
        while (k-- > 0) {
            Integer vv = que.poll();
            que.add(vv + 1);
        }
        long ans = 1;
        while (!que.isEmpty()) {
            ans *= que.poll();
            ans %= 1000000007;
        }
        return (int) ans;
    }

    public long maximumBeauty(int[] flowers, long newFlowers, int target, int full, int partial) {
        Arrays.sort(flowers);
        LinkedList<Integer> list = new LinkedList<>();
        for (int i = 0;i < flowers.length; i++) {
            list.add(flowers[i]);
        }
        long cnt = 0;
        while (!list.isEmpty() && list.getLast() >= target) {
            list.removeLast();
            cnt++;
        }
        if (list.isEmpty()) {
            return cnt * full;
        }
        long res = 0, sum = 0;
        for (int i = 0;i < list.size(); i++) {
            sum += list.get(i);
        }
        long T = target - 1;
        for (int i = list.size(), j = i - 1;i >= 0; i--, cnt++) {
            if (i < list.size()) {
                newFlowers -= target - list.get(i);
            }
            if (newFlowers < 0) {
                break;
            }
            if (i > 0) {
                while (j >= i) {
                    sum -= list.get(j);
                    j--;
                }
                while (T * (j + 1) - sum > newFlowers) {
                    T--;
                    while (flowers[j] >= T) {
                        sum -= flowers[j];
                        j--;
                    }
                }
                res = Math.max(res, T* partial + cnt * full);
            } else {
                res = Math.max(res, cnt * full);
            }
        }
        return res;
    }

    public int countNumbersWithUniqueDigits(int n) {
        if (n == 0) {
            return 1;
        }
        if (n == 1) {
            return 10;
        }
        int res = 10, cur = 9;
        for (int i = 0;i < n - 1; i++) {
            cur *= 9;
            res += cur;
        }
        return res;
    }

    class RandomizedSet {

        Set<Integer> set;

        public RandomizedSet() {
            set = new HashSet<>();
        }

        public boolean insert(int val) {
            if (set.contains(val)) {
                return false;
            }
            set.add(val);
            return true;
        }

        public boolean remove(int val) {
            set.remove(val);
            return false;
        }

        public int getRandom() {
            Random random = new Random();
            int a = random.nextInt(set.size());
            for (int x : set) {
                if (a == x) {
                    return a;
                }
            }
            return 0;
        }
    }

    public int giveGem(int[] gem, int[][] operations) {
        for (int[] arr : operations) {
            int x = arr[0];
            int y = arr[1];
            gem[y] += gem[x] / 2;
            gem[x] -= gem[x] / 2;
        }
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        for (int i = 0;i < gem.length; i++) {
            max = Math.max(max, gem[i]);
            min = Math.max(min, gem[i]);
        }
        return max - min;
    }

    public int perfectMenu(int[] materials, int[][] cookbooks, int[][] attribute, int limit) {
        int n = cookbooks.length;
        int ans = -1;
        for (int i = 1;i < (1 << n); i++) {
            int cur = i;
            int vv = 0;
            int vv1 = 0;
            int[] arr = new int[5];
            for (int j = 0;j < 5; j++) {
                arr[j] = materials[j];
            }
            for (int j = 0;j < n; j++) {
                if ((cur & 1) == 1) {
                    boolean flag = true;
                    for (int k = 0;k < 5; k++) {
                        if (arr[k] - cookbooks[j][k] >= 0) {
                            arr[k] -= cookbooks[j][k];
                        } else {
                            flag = false;
                            break;
                        }
                    }
                    if (!flag) {
                        break;
                    }
                    vv += attribute[j][0];
                    vv1 += attribute[j][1];
                }
                cur >>= 1;
            }
            if (vv1 >= limit) {
                ans = Math.max(ans, vv);
            }
        }
        return ans;
    }

    private int[] s = new int[505050];
    private int[] col = new int[505050];
    private List<Integer> vl = new ArrayList<>();

    public void down(int p, int l, int r) {
        if(col[p]!=0) {
            int mid=(l+r)/2;
            s[p*2]=col[p]*(mid-l+1);
            s[p*2+1]=col[p]*(r-mid);
            col[p*2]=col[p*2+1]=col[p];
            col[p]=0;
        }
    }

    public void up(int p) {
        s[p]=s[p*2]+s[p*2+1];
    }

    public void renew(int p,int l,int r,int x,int y,int v)
    {
        if(x<=l&&y>=r)
        {
            s[p]=(r-l+1)*v;
            col[p]=v;
            return;
        }
        down(p,l,r);
        int mid=(l+r)/2;
        if(x<=mid) renew(p*2,l,mid,x,y,v);
        if(y>mid) renew(p*2+1,mid+1,r,x,y,v);
        up(p);
    }

    public int getNumber(TreeNode root, int[][] ops) {
        preRoot(root);
        int n = vl.size();
        Collections.sort(vl);
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0;i < n; i++) {
            map.put(vl.get(i), i);
        }
        renew(1,1,n,1,n,0);
        for (int[] op : ops) {
            int x = op[0];
            int y = op[1];
            int z = op[2];
            int sv = map.get(y) + 1;
            int e = map.get(z) + 1;
            if (x == 0) {
                renew(1,1,n,sv,e,0);
            } else {
                renew(1,1,n,sv,e,1);
            }
        }
        return s[1];
    }

    public void preRoot(TreeNode root) {
        if (root == null) {
            return ;
        }
        preRoot(root.left);
        vl.add(root.val);
        preRoot(root.right);
    }

    public long waysToBuyPensPencils(int total, int cost1, int cost2) {
        long ans = 0;
        int index = 0;
        while (true) {
            int vv = total;
            vv -= index * cost1;
            if (vv < 0) {
                break;
            }
            ans += vv / cost2 + 1;
        }
        return  ans;
    }

    class ATM {

        long[] arr = new long[5];

        public ATM() {

        }

        public void deposit(int[] banknotesCount) {
            for (int i = 0;i < 5; i++) {
                arr[i] += banknotesCount[i];
            }
        }

        public int[] withdraw(int amount) {
            int[] arr1 = new int[5];
            if (arr1[4] >= amount / 500) {
                int cur = amount / 500;
                amount -= 500 * cur;
                arr1[4] = cur;
            }
            if (arr1[3] >= amount / 200) {
                int cur = amount / 200;
                amount -= cur * 200;
                arr1[3] = cur;
            }
            if (arr1[2] >= amount / 100) {
                int cur = amount / 100;
                amount -= cur * 100;
                arr1[2] = cur;
            }
            if (arr1[1] >= amount / 50) {
                int cur = amount / 50;
                amount -= cur * 50;
                arr1[1] = cur;
            }
            if (arr1[0] >= amount / 20) {
                int cur = amount / 20;
                amount -= cur * 20;
                arr1[0] = cur;
            }
            if (amount == 0) {
                for (int i = 0;i < 5; i++) {
                    arr[i] -= arr1[i];
                }
                return arr1;
            } else {
                return new int[]{-1};
            }
        }
    }

    public int maximumScore(int[] scores, int[][] edges) {
        int n = scores.length;
        List<int[]>[] g = new ArrayList[n];
        for (int i = 0;i < n; i++) {
            g[i] = new ArrayList<>();
        }
        for (int[] edge : edges) {
            int x = edge[0];
            int y = edge[1];
            g[x].add(new int[]{scores[y], y});
            g[y].add(new int[]{scores[x], x});
        }
        for (int i = 0;i < n; i++) {
            if (g[i].size() > 3) {
                g[i].sort((a, b) -> (b[0] - a[0]));
                g[i] = new ArrayList<>(g[i].subList(0, 3));
            }
        }
        int ans = -1;
        for (int[] edge : edges) {
            int x = edge[0], y = edge[1];
            for (int[] p : g[x]) {
                int a = p[1];
                for (int [] q : g[y]) {
                    int b = q[1];
                    if (a != y && b != x && a != b) {
                        ans = Math.max(ans, p[0] + q[0] + scores[x] + scores[y]);
                    }
                }
            }
        }
        return ans;
    }

    public String digitSum(String s, int k) {
        StringBuilder sb = new StringBuilder();
        sb.append(s);
        while (sb.length() > k) {
            StringBuilder sb1 = new StringBuilder();
            for (int i = 0;i < sb.length(); i+= k) {
                String s1 = sb.substring(i, Math.min(i + k, sb.length()));
                int cur = 0;
                for (int j = 0;j < s1.length(); j++) {
                    cur += s1.charAt(j) - '0';
                }
                sb1.append(cur);
            }
            sb = sb1;
        }
        return sb.toString();
    }

    public int minimumRounds(int[] tasks) {
        Map<Integer, Integer> ma = new HashMap<>();
        for (int x : tasks) {
            ma.put(x, ma.getOrDefault(x, 0) + 1);
        }
        int ans = 0;
        for (int x : ma.keySet()) {
            int cur = ma.get(x);
            if (cur % 3 == 0) {
                ans += cur / 3;
                continue;
            }
                int vv = 1;
                while (true) {
                    if ((cur - 2 * vv) % 3 == 0 || (cur - 2 * vv < 0)) {
                        break;
                    }
                    vv++;
                }
                if (cur - 2 * vv < 0) {
                    ans = -1;
                    break;
                }
                ans += vv + (cur - 2 * vv) / 3;
        }
        return ans;
    }

    public int maxTrailingZeros(int[][] grid) {
        int n = grid.length, m = grid[0].length;
        int[][] f2 = new int[n + 1][m + 1];
        int[][] g2 = new int[n + 1][m + 1];
        int[][] f5 = new int[n + 1][m + 1];
        int[][] g5 = new int[n + 1][m + 1];
        for (int i = 1;i <= n; i++) {
            for (int j = 1;j <= m; j++) {
                int x = grid[i - 1][j - 1];
                int two = 0, five = 0;
                while (x % 2 == 0) {
                    two++;
                    x /= 2;
                }
                while (x % 5 == 0) {
                    five++;
                    x /= 5;
                }
                f2[i][j] = f2[i][j - 1] + two;
                g2[i][j] = g2[i - 1][j] + two;
                f5[i][j] = f5[i][j - 1] + five;
                g5[i][j] = g5[i - 1][j] + five;
            }
        }
        int ans = 0;
        for (int i = 1;i <= n; i++) {
            for (int j = 1;j <= m; j++) {
                ans = Math.max(ans, Math.min(f2[i][j] + g2[i - 1][j], f5[i][j] + g5[i - 1][j]));
                ans = Math.max(ans, Math.min(f2[i][j] + g2[n][j] - g2[i][j], f5[i][j] + g5[n][j] - g5[i][j]));
                ans = Math.max(ans, Math.min(f2[i][m] - f2[i][j] + g2[i][j], f5[i][m] - f5[i][j] + g5[i][j]));
                ans = Math.max(ans, Math.min(f2[i][m] - f2[i][j - 1] + g2[n][j] - g2[i][j],
                                            f5[i][m] - f5[i][j - 1] + g5[n][j] - g5[i][j]));
            }
        }
        return ans;
    }

    List<Integer> vvl = new ArrayList<>();
    public List<Integer> lexicalOrder(int n) {
        for (int i = 1;i <= 9; i++) {
            if (i <= n) {
                vvl.add(i);
                dfs(i, n);
            }
        }
        return vvl;
    }

    public void dfs(int k, int n) {
        for (int i = 0;i < 10; i++) {
            int cur = k * 10 + i;
            if (cur <= n) {
                vvl.add(cur);
                dfs(cur, n);
            }
        }
    }

    public int lengthLongestPath(String input) {
        int n = input.length();
        int pos = 0;
        int ans = 0;
        Deque<Integer> que = new ArrayDeque<>();
        while (pos < n) {
            int depth = 1;
            while (pos < n && input.charAt(pos) == '\t') {
                pos++;
                depth++;
            }
            boolean isFile = false;
            int cur = 0;
            while (pos < n && input.charAt(pos) != '\n') {
                if (input.charAt(pos) == '.') {
                    isFile = true;
                }
                cur++;
                pos++;
            }
            pos++;

            while (que.size() >= depth) {
                que.poll();
            }
            if (!que.isEmpty()) {
                cur += que.peek() + 1;
            }
            if (isFile) {
                ans = Math.max(cur, ans);
            } else {
                que.push(cur);
            }
        }
        return ans;
    }

    public int getMinimumTime(int[] time, int[][] fruits, int limit) {
        int ans = 0;
        for (int i = 0;i < fruits.length; i++) {
            int type = fruits[i][0];
            int num = fruits[i][1];
            ans += (num % limit == 0 ? num / limit : num / limit + 1) * time[type];
        }
        return ans;
    }

    private String vv = "<>v^";
    private int[][] dir = {
            {0,-1},{0,1},{1,0},{-1,0}
    };

    public int conveyorBelt(String[] matrix, int[] start, int[] end) {
        int n = matrix.length;
        int m = matrix[0].length();
        char[][] chars = new char[105][105];
        int[][] dis = new int[105][105];
        for (int i = 0;i < n; i++) {
            for (int j = 0;j < m; j++) {
                chars[i][j] = matrix[i].charAt(j);
            }
        }
        Queue<int[]> que = new LinkedList<>();
        que.add(start);
        while (!que.isEmpty()) {
            int[] curArr = que.poll();
            int x = curArr[0], y = curArr[1];
            int index = 0;
            for (int i = 0;i < 4; i++) {
                if (chars[x][y] == vv.charAt(i)) {
                    index = i;
                    break;
                }
            }
            for (int i = 0;i < 4; i++) {
                int nx = x + dir[i][0];
                int ny = y + dir[i][1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < m) {
                    int cur = index != i ? 1 : 0;
                    if (dis[nx][ny] > (dis[x][y] + cur)) {
                        dis[nx][ny] = dis[x][y] + cur;
                        que.add(new int[]{x, y});
                    }
                }
            }
        }
        return dis[end[0]][end[1]];
    }

    public List<Integer> intersection(int[][] nums) {
        int n = nums.length;
        int m = nums[0].length;
        int[] arr = new int[1010];
        for (int i = 0;i < n; i++) {
            for (int j = 0;j < m; j++) {
                arr[nums[i][j]]++;
            }
        }
        List<Integer> list = new ArrayList<>();
        for (int i = 1;i <= 1000; i++) {
            if (arr[i] == nums.length)
                list.add(arr[i]);
        }
        return list;
    }

    public int countLatticePoints(int[][] circles) {
        Set<String> set = new HashSet<>();
        for (int i = 0;i < circles.length; i++) {
            int x = circles[i][0];
            int y = circles[i][1];
            int r = circles[i][2];
            for (int j = x - r;j <= x + r; j++) {
                for (int k = y - r;k <= y + r; k++) {
                    if (distance(x, y, j, k) <= r * 1.0) {
                        set.add("" + j + k);
                    }
                }
            }
        }
        return set.size();
    }

    public double distance(int x1, int y1, int x2, int y2) {
        return Math.sqrt(1.0 * (y2 - y1) * (y2 - y1) + (x2 - x1) * (x2- x1));
    }

    public int[] countRectangles(int[][] rectangles, int[][] points) {
        Arrays.sort(rectangles, (a, b) -> b[1] - a[1]);
        int n = points.length;
        Integer[] ids = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, (i, j) -> (points[j][1] - points[i][1]));
        int[] ans = new int[n];
        int i = 0;
        List<Integer> list = new ArrayList<>();
        for (int id : ids) {
            int start = i;
            while (i < rectangles.length && rectangles[i][1] >= points[id][1]) {
                list.add(rectangles[i++][0]);
            }
            if (start < i) {
                Collections.sort(list);
            }
            ans[id] =  i - lower_bound(list, points[id][0]);
        }
        return ans;
    }

    private int lower_bound(List<Integer> list, int key) {
        int l = 0, r = list.size();
        while (l < r) {
            int mid = (l + r) >> 1;
            if (list.get(mid) >= key) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    public int[] fullBloomFlowers(int[][] flowers, int[] persons) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int[] f : flowers) {
            map.put(f[0], map.getOrDefault(f[0], 0) + 1);
            map.put(f[1] + 1, map.getOrDefault(f[1] + 1, 0) - 1);
        }
        int n = persons.length;
        int index = 0;
        int sum = 0;
        int[] ans = new int[n];
        int[] times = map.keySet().stream().mapToInt(Integer::intValue).sorted().toArray();
        Integer[] ids = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, Comparator.comparingInt(i -> persons[i]));
        for (int id : ids) {
            while (index < times.length && persons[id] >= times[index]) {
                sum += map.get(times[index++]);
            }
            ans[id] = sum;
        }
        return ans;
    }

    public int countPrefixes(String[] words, String s) {
        int cur = 0;
        for (int i = 0;i < words.length; i++) {
            if (s.startsWith(words[i])) {
                cur++;
            }
        }
        return cur;
    }

    public int minimumAverageDifference(int[] nums) {
        int n = nums.length;
        int[] sum  = new int[n + 1];
        for (int i = 0;i < n; i++) {
            if (i == 0) {
                sum[i + 1] = nums[i];
            } else {
                sum[i + 1] = sum[i] + nums[i];
            }
        }
        int index = 0;
        int min = Integer.MAX_VALUE;
        for (int i = 1;i <= n; i++) {
            int cur1 = sum[i] / i;
            int cur2;
            if (n - i != 0) {
                cur2 = (sum[n] - sum[i]) / (n - i);
            } else {
                cur2=0;
            }
            if (Math.abs(cur1 - cur2) < min) {
                min = Math.abs(cur1 - cur2);
                index = i - 1;
            }
        }
        return index;
    }

    public int countUnguarded(int m, int n, int[][] guards, int[][] walls) {
        Guard[][] guardv = new Guard[m][n];
        for (int i = 0;i < m; i++) {
            for (int j = 0;j < n; j++) {
                guardv[i][j] = new Guard(0, false, false, false, false);
            }
        }
        for (int[] w : walls) {
            guardv[w[0]][w[1]].val = 3;
        }
        for (int[] g : guards) {
            guardv[g[0]][g[1]].val = 2;
        }
        for (int[] g : guards) {
            int x = g[0];
            int y = g[1];
            if (!guardv[x][y].l) {
                for (int i = y - 1;i >= 0; i--) {
                    if (guardv[x][i].val == 0) {
                        guardv[x][i].val = 1;
                    }
                    if (guardv[x][i].val == 2) {
                        guardv[x][i].r = true;
                        break;
                    }
                    if (guardv[x][i].val == 3) {
                        break;
                    }
                }
            }
            if (!guardv[x][y].r) {
                for (int i = y + 1;i < n; i++) {
                    if (guardv[x][i].val == 0) {
                        guardv[x][i].val = 1;
                    }
                    if (guardv[x][i].val == 2) {
                        guardv[x][i].l = true;
                        break;
                    }
                    if (guardv[x][i].val == 3) {
                        break;
                    }
                }
            }
            if (!guardv[x][y].d) {
                for (int i = x + 1;i < m; i++) {
                    if (guardv[i][y].val == 0) {
                        guardv[i][y].val = 1;
                    }
                    if (guardv[i][y].val == 2) {
                        guardv[i][y].u = true;
                        break;
                    }
                    if (guardv[i][y].val == 3) {
                        break;
                    }
                }
            }
            if (!guardv[x][y].u) {
                for (int i = x - 1;i >= 0; i--) {
                    if (guardv[i][y].val == 0) {
                        guardv[i][y].val = 1;
                    }
                    if (guardv[i][y].val == 2) {
                        guardv[i][y].d = true;
                        break;
                    }
                    if (guardv[i][y].val == 3) {
                        break;
                    }
                }
            }
        }
        int ans = 0;
        for (int i = 0;i < m; i++) {
            for (int j = 0;j < n; j++) {
                if (guardv[i][j].val == 0) {
                    ans++;
                }
            }
        }
        return ans;
    }

    class Guard {
        int val;
        boolean l;
        boolean r;
        boolean d;
        boolean u;

        public Guard(int val, boolean l, boolean r, boolean d, boolean u) {
            this.val = val;
            this.l = l;
            this.r = r;
            this.d = d;
            this.u = u;
        }
    }

    public String removeDigit(String number, char digit) {
        List<String> list = new ArrayList<>();
        LinkedList<Character> l = new LinkedList<>();
        for (int i = 0;i < number.length(); i++) {
            l.add(number.charAt(i));
        }
        for (int i = 0;i < number.length(); i++) {
            if (number.charAt(i) == digit) {
                l.remove(i);
                StringBuilder sb = new StringBuilder();
                for (int j = 0;j < l.size(); j++) {
                    sb.append(l.get(j));
                }
                list.add(sb.toString());
                l.add(i, number.charAt(i));
            }
        }
        Collections.sort(list);
        return list.get(list.size() - 1);
    }

    public int minimumCardPickup(int[] cards) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0;i < cards.length; i++) {
            map.putIfAbsent(cards[i], new ArrayList<>());
            map.get(cards[i]).add(i);
        }
        int ans = -1;
        int cur = Integer.MAX_VALUE;
        for (int x :map.keySet()) {
            if (map.get(x).size() > 1) {
                for (int i = 1;i < map.get(x).size(); i++) {
                    cur = Math.min(cur, map.get(x).get(i) - map.get(x).get(i - 1) + 1);
                    ans = 1;
                }
            }
        }
        if (ans == -1) {
            return -1;
        }
        return cur;
    }

    public int countDistinct(int[] nums, int k, int p) {
        int ans = 0;
        Set<String> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            int cur = 0;
            StringBuilder sb = new StringBuilder();
            for (int j = i;j < nums.length; j++) {
                if (nums[j] % p == 0) {
                    cur++;
                }
                sb.append(nums[j] + ":");
                if (set.contains(sb.toString())) {
                    continue;
                }
                set.add(sb.toString());
                if (cur >= 0 && cur <= k) {
                    ans++;
                }
            }
        }
        return ans;
    }

    public long appealSum(String s) {
        int[] pos = new int[26];
        Arrays.fill(pos, -1);
        long ans = 0;
        long sum = 0;
        for (int i = 0;i < s.length(); i++) {
            int v = s.charAt(i) - 'a';
            sum += i - pos[v];
            ans += sum;
            pos[v] = i;
        }
        return ans;
    }

    static int[][] dir1 = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    public int maximumMinutes(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int l = -1, r = m * n;
        while (l < r) {
            int mid = (l + r + 1) >> 1;
            if (check(grid, mid)) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        return l < m * n ? l : (int) 1e9;
    }

    boolean check(int[][] grid, int t) {
        int m = grid.length, n = grid[0].length;
        boolean[][] fire = new boolean[m][n];
        ArrayList<int[]> f = new ArrayList<>();
        for (int i = 0;i < m; i++) {
            for (int j = 0;j < n; j++) {
                if (grid[i][j] == 1) {
                    fire[i][j] = true;
                    f.add(new int[]{i, j});
                }
            }
        }
        while (t-- > 0 && f.size() > 0) {
            f = spreadFire(grid, fire, f);
        }
        if (fire[0][0]) {
            return false;
        }
        boolean[][] vis = new boolean[m][n];
        vis[0][0] = true;
        ArrayList<int[]> q = new ArrayList<>();
        q.add(new int[]{0, 0});
        while (q.size() > 0) {
            ArrayList<int[]> tmp = q;
            q = new ArrayList<>();
            for (int[] p : tmp) {
                if (!fire[p[0]][p[1]]) {
                    for (int[] d : dir1) {
                        int nx = p[0] + d[0];
                        int ny = p[1] + d[1];
                        if (nx >= 0 && nx < m && ny >= 0 && ny < n
                                && !fire[nx][ny] && !vis[nx][ny]
                                && grid[nx][ny] != 2) {
                            if (nx == m - 1 && ny == n - 1) {
                                return true;
                            }
                            vis[nx][ny] = true;
                            q.add(new int[]{nx, ny});
                        }
                    }
                }
            }
            f = spreadFire(grid, fire, f);
        }
        return false;
    }

    ArrayList<int[]> spreadFire(int[][] grid, boolean[][] fire, ArrayList<int[]> f) {
        int m = grid.length, n = grid[0].length;
        List<int[]> tmp = f;
        f = new ArrayList<>();
        for (int[] ins : tmp) {
            for (int[] d : dir1) {
                int nx = ins[0] + d[0];
                int ny = ins[1] + d[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n
                        && !fire[nx][ny]
                        && grid[nx][ny] != 2) {
                    fire[nx][ny] = true;
                    f.add(new int[] {nx, ny});
                }
            }
        }
        return f;
    }

    public boolean isValid(String code) {
        int n = code.length();
        Deque<String> que = new ArrayDeque<>();
        int i = 0;
        while (i < n) {
            if (code.charAt(i) == '<') {
                if (i == n - 1) {
                    return false;
                }
                if (code.charAt(i + 1) == '/') {
                    int j = code.indexOf('>', i);
                    if (j < 0) {
                        return false;
                    }
                    String tagName = code.substring(i + 2, j);
                    if (que.isEmpty() || !que.peek().equals(tagName)) {
                        return false;
                    }
                    que.pop();
                    i = j + 1;
                    if (que.isEmpty() && i != n) {
                        return false;
                    }
                } else if (code.charAt(i + 1) == '!') {
                    if (que.isEmpty()) {
                        return false;
                    }
                    if (i + 9 >= n) {
                        return false;
                    }
                    String cur = code.substring(i + 2, i + 9);
                    if (!cur.equals("[CDATA")) {
                        return false;
                    }
                    int j = code.indexOf("]]>", i);
                    if (j < 0) {
                        return false;
                    }
                    i = j + 1;
                } else {
                    int j = code.indexOf('>', i);
                    if (j < 0) {
                        return false;
                    }
                    String tagName = code.substring(i + 1, j);
                    if (tagName.length() < 1 || tagName.length() > 9) {
                        return false;
                    }
                    for (int v = 0;v < tagName.length(); v++) {
                         if (!Character.isUpperCase(tagName.charAt(v))) {
                             return false;
                         }
                    }
                    que.push(tagName);
                    i = j + 1;
                }
            } else {
                if (que.isEmpty()) {
                    return false;
                }
                i++;
            }
        }
        return que.isEmpty();
    }

    private int ansab = 0;

    public int averageOfSubtree(TreeNode root) {
        backOrder(root);
        return ansab;
    }

    public Pair backOrder(TreeNode root) {
        if (root == null) {
            return new Pair(0, 0);
        }
        Pair cur1 = backOrder(root.left);
        Pair cur2 = backOrder(root.right);
        int c1 = cur1.x + cur2.x + root.val;
        int c2 = cur1.y + cur2.y + 1;
        if (c1 / c2 == root.val) {
            ansab++;
        }
        return new Pair(c1, c2);
    }

    public int countTexts(String pressedKeys) {
        int n = pressedKeys.length();
        int mod = 1000000007;
        long[] dp1 = new long[n + 1];
        long[] dp2 = new long[n + 1];
        dp1[0] = 1;
        dp1[1] = 2;
        dp1[2] = 4;
        dp2[0] = 1;
        dp2[1] = 1;
        dp2[2] = 2;
        dp2[3] = 4;
        for (int i = 3;i <= n; i++) {
            dp1[i] = dp1[i - 1] + dp1[i - 2] + dp1[i - 3];
            if (i != 3) {
                dp2[i] = dp2[i - 1] + dp2[i - 2] + dp2[i - 3] + dp2[i - 4];
            }
        }
        char cur = pressedKeys.charAt(0);
        int v = 0;
        long ans = 1;
        for (int i = 0;i < n; i++) {
            char vv = pressedKeys.charAt(i);
            if (cur != vv) {
                if (cur == '7' || cur == '9') {
                    ans *= dp2[v - 1];
                } else {
                    ans *= dp1[v - 1];
                }
                ans %= mod;
                cur = vv;
                v = 0;
            }
            v++;
        }
        if (cur == '7' || cur == '9') {
            ans *= dp2[v - 1];
        } else {
            ans *= dp1[v - 1];
        }
        ans %= mod;
        return (int) ans;
    }

    public static void main(String[] args) throws NoSuchAlgorithmException {

        new Main().countTexts("444479999555588866");
    }

    static class Node {
        public int val;
        public List<Node> children;

        public Node() {}

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, List<Node> _children) {
            val = _val;
            children = _children;
        }
    };

    class HeapNode {
        int u;
        long val;

        public HeapNode(int u, long val) {
            this.u = u;
            this.val = val;
        }
    }
}


