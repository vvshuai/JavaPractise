package com.vvs.leetcode;

import com.vvs.jvm0609.T;
import javafx.util.Pair;
import org.apache.rocketmq.client.QueryResult;
import sun.reflect.generics.tree.Tree;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;
import java.util.stream.IntStream;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 20:11 2022/5/8
 * @Modified By:
 */
public class NewMain {

    private int m, n;
    private boolean[][][] vis;
    private char[][] grid;

    public boolean hasValidPath(char[][] grid) {
        this.m = grid.length;
        this.n = grid[0].length;
        if ((m + n - 1) % 2 != 0) {
            return false;
        }
        if (grid[0][0] == ')' || grid[0][0] == '(') {
            return false;
        }
        this.grid = grid;
        this.vis = new boolean[m][n][(m + n + 1) / 2];
        return dfs(0, 0, 0);
    }

    public boolean dfs(int x, int y, int c) {
        if (c > m - x + n - y - 1) {
            return false;
        }
        if (x == m - 1 && y == n - 1) {
            return true;
        }
        if (vis[x][y][c]) {
            return false;
        }
        vis[x][y][c] = true;
        c += grid[x][y] == '(' ? 1 : -1;
        return c >= 0 && (x < m - 1 && dfs(x + 1, y , c)) &&(y < n - 1 && dfs(x, y + 1, c));
    }

    public int[] diStringMatch(String s) {
        int[] ans = new int[s.length() + 1];
        int l = 0, r = s.length();
        for (int i = 0;i < s.length(); i++) {
            if (s.charAt(i) == 'I') {
                ans[i] = l++;
            } else {
                ans[i] = r--;
            }
        }
        ans[s.length()] = l;
        return ans;
    }

    public String serialize(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        postOrder(root, list);
        String str = list.toString();
        return str.substring(1, str.length() - 1);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.isEmpty()) {
            return null;
        }
        String[] strings = data.split(", ");
        Deque<Integer> stack = new ArrayDeque<>();
        int len = strings.length;
        for (int i = 0;i < len; i++) {
            stack.push(Integer.parseInt(strings[i]));
        }
        return construct(Integer.MIN_VALUE, Integer.MAX_VALUE, stack);
    }

    private TreeNode construct(int minValue, int maxValue, Deque<Integer> stack) {
        if (stack.isEmpty() || stack.peek() < minValue || stack.peek() > maxValue) {
            return null;
        }
        int val = stack.pop();
        TreeNode root = new TreeNode(val);
        root.right = construct(val, maxValue, stack);
        root.left = construct(minValue, val, stack);
        return root;
    }

    private void postOrder(TreeNode node, List<Integer> list) {
        if (node == null) {
            return ;
        }
        postOrder(node.left, list);
        postOrder(node.right, list);
        list.add(node.val);
    }

    public boolean oneEditAway(String first, String second) {
        if (first.equals(second)) {
            return true;
        }
        int m = first.length();
        int n = second.length();
        if (m == n) {
            boolean flag = true;
            for (int i = 0;i < n; i++) {
                if (first.charAt(i) != second.charAt(i) && flag) {
                    flag = false;
                }
                if (first.charAt(i) != second.charAt(i) && !flag) {
                    return false;
                }
            }
        } else if (Math.abs(m - n) == 1) {
            if (m > n) {
                return OneInsert(first, second);
            } else {
                return OneInsert(second, first);
            }
        }
        return false;
    }

    private boolean OneInsert(String first, String second) {
        int m = first.length();
        int n = second.length();
        int index1 = 0;
        int vv  = 0;
        for (int i = 0;i < m; i++) {
            if (first.charAt(i) == second.charAt(index1)) {
                index1++;
            } else {
                vv++;
            }
            if (vv > 1) {
                return false;
            }
        }
        return true;
    }

    public int divisorSubstrings(int num, int k) {
        String s = String.valueOf(num);
        int ans = 0;
        for (int i = 0;i < n; i++) {
            String cur = s.substring(i, i + k);
            int vv = Integer.parseInt(cur);
            if (num % vv == 0){
                ans++;
            }
        }
        return ans;
    }

    public int waysToSplitArray(int[] nums) {
        int n = nums.length;
        int[] sum = new int[n + 1];
        for (int i = 0;i < nums.length; i++) {
            sum[i + 1] = sum[i] + nums[i];
        }
        int ans = 0;
        for (int i = 1;i < n; i++) {
            if (sum[i] > sum[n] - sum[i]) {
                ans++;
            }
        }
        return ans;
    }

    public int maximumWhiteTiles(int[][] tiles, int carpetLen) {
        Arrays.sort(tiles, Comparator.comparingInt(o -> o[0]));
        LinkedList<int[]> list = new LinkedList<>();
        List<Integer> vvList = new ArrayList<>();
        int[] curArr = new int[tiles.length + 1];
        for (int[] tile : tiles) {
            if (list.isEmpty()) {
                list.add(tile);
            } else {
                int[] last = list.getLast();
                if (last[1] + 1 == tile[0]) {
                    int[] newTile = new int[2];
                    newTile[0] = last[0];
                    newTile[1] = tile[1];
                    list.pollLast();
                    list.add(newTile);
                } else {
                    list.add(tile);
                }
            }
        }
        List<int[]> newList = new ArrayList<>(list);
        for (int i = 0;i < newList.size(); i++) {
            vvList.add(newList.get(i)[0]);
            curArr[i + 1] = curArr[i] + (newList.get(i)[1] - newList.get(i)[0] + 1);
        }
        int n = newList.size();
        int max = 0;
        for (int i = 0;i < newList.size(); i++) {
            int[] tile = newList.get(i);
            long cur = carpetLen + tile[0] - 1;
            if (cur >= 1000000000) {
                max = Math.max(max, curArr[n] - curArr[i]);
            } else {
                int vv = upper_bound(vvList, (int) cur);
                int[] last = newList.get(vv - 1);
                if (cur >= last[1]) {
                    max = Math.max(max, curArr[vv] - curArr[i]);
                } else {
                    max = (int) Math.max(max, curArr[vv - 1] - curArr[i] + (cur - last[0]));
                }
            }
        }
        return max;
    }

    public int upper_bound(List<Integer> list, int target) {
        int l = 0;
        int r = list.size() - 1;
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

    public List<String> removeAnagrams(String[] words) {
        Deque<String> stack = new ArrayDeque<>();
        for (int i = 0;i < words.length; i++) {
            if (stack.isEmpty()) {
                stack.push(words[i]);
            } else {
                String last = stack.peek();
                char[] chars1 = last.toCharArray();
                char[] chars2 = words[i].toCharArray();
                Arrays.sort(chars1);
                Arrays.sort(chars2);
                boolean flag = true;
                if (chars1.length == chars2.length) {
                    for (int j = 0;j < chars1.length; j++) {
                        if (chars1[j] != chars2[j]) {
                            flag = false;
                            break;
                        }
                    }
                } else {
                    flag = false;
                }
                if (flag) {
                    continue;
                }
                stack.push(words[i]);
            }
        }
        List<String> ans = new ArrayList<>();
        while (!stack.isEmpty()) {
            ans.add(stack.pollFirst());
        }
        return ans;
    }

    public int maxConsecutive(int bottom, int top, int[] special) {
        int n = special.length;
        int[] curArr = new int[n + 2];
        for (int i = 0;i < n; i++) {
            curArr[i] = special[i];
        }
        curArr[n] = bottom;
        curArr[n + 1] = top;
        Arrays.sort(curArr);
        int ans = 0;
        for (int i = 1;i < n + 2; i++) {
            int vv = (i == 1 || i == n + 1) ? 0 : 1;
            ans = Math.max(ans, curArr[i] - curArr[i - 1] - vv);
        }
        return ans;
    }

    public int largestCombination(int[] candidates) {
        int[] cur = new int[32];
        for (int i = 0;i < n; i++) {
            int v = candidates.length;
            int c = 0;
            while (v > 0) {
                if (v % 2 == 1) {
                    cur[c]++;
                }
                c++;
                v >>= 1;
            }
        }
        int max = 0;
        for (int i = 0;i < 32; i++) {
            max = Math.max(cur[i], max);
        }
        return max;
    }

    public int largestVariance(String s) {
        int n = s.length();
        int ans = 0;
        for (char a = 'a';a <= 'z'; a++) {
            for (char b = 'a';b <= 'z'; b++) {
                if (a == b) {
                    continue;
                }
                int diff = 0;
                int diffB = -n;
                for (int i = 0;i < s.length(); i++) {
                    if (s.charAt(i) == a) {
                        diff++;
                        diffB++;
                    } else if (s.charAt(i) == b) {
                        diffB = --diff;
                        diffB = Math.max(diffB, 0);
                    }
                    ans = Math.max(ans, diffB);
                }
            }
        }
        return ans;
    }

    public int[] findRightInterval(int[][] intervals) {
        int n = intervals.length;
        int[] ans = new int[n];
        InterVal[] vals = new InterVal[n];
        for (int i = 0;i < n; i++) {
            vals[i] = new InterVal(i, intervals[i]);
        }
        Arrays.sort(vals, Comparator.comparingInt(a -> a.val[0]));
        for (int i = 0;i < n; i++) {
            int end = vals[i].val[1];
            int l = 0, r = n;
            while (l < r) {
                int mid = (l + r) >> 1;
                if (vals[mid].val[0] < end) {
                    l = mid + 1;
                } else {
                    r = mid;
                }
            }
            if (l == i || l >= n) {
                ans[vals[i].index] = -1;
                continue;
            }
            ans[vals[i].index] = l;
        }
        return ans;
    }

    class InterVal {
        int index;
        int[] val;

        public InterVal(int index, int[] val) {
            this.index = index;
            this.val = val;
        }
    }

    public int findKthNumber(int m, int n, int k) {
        int l = 1;
        int r = m * n;
        while (l < r) {
            int mid = (l + r) >> 1;
            if (count(m, n, mid) >= k) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    public int count(int m, int n, int k) {
        int res = 0;
        for (int i = 0;i < m; i++) {
            res += Math.min(n, k / i);
        }
        return res;
    }

    public int percentageLetter(String s, char letter) {
        int n = s.length();
        int cur = 0;
        for (int i = 0;i < n; i++) {
            if (s.charAt(i) == letter) {
                cur++;
            }
        }
        return cur / n;
    }

    public int maximumBags(int[] capacity, int[] rocks, int additionalRocks) {
        int n = capacity.length;
        int[][] arr = new int[n][2];
        for (int i = 0;i < n; i++) {
            arr[i] = new int[]{i, capacity[i] - rocks[i]};
        }
        Arrays.sort(arr, (a, b) -> a[1]-a[1]);
        int ans = 0;
        for (int i = 0;i < n; i++) {
            if (additionalRocks - arr[i][1] >= 0) {
                ans++;
                additionalRocks -= arr[i][1];
            } else {
                break;
            }
        }
        return ans;
    }

    public int minimumLines(int[][] stockPrices) {
        int ans = 0;
        Arrays.sort(stockPrices, Comparator.comparingInt(a -> a[0]));
        BigDecimal lastk = new BigDecimal(0);
        for (int i = 1;i < stockPrices.length; i++) {
            int[] cur = stockPrices[i];
            int[] last = stockPrices[i - 1];
            BigDecimal x1 = new BigDecimal(String.valueOf(last[0]));
            BigDecimal y1 = new BigDecimal(String.valueOf(last[1]));
            BigDecimal x2 = new BigDecimal(String.valueOf(cur[0]));
            BigDecimal y2 = new BigDecimal(String.valueOf(cur[1]));
            BigDecimal k;
            if (x1.equals(x2)) {
                k = new BigDecimal(1);
            } else if (y1.equals(y2)) {
                k = new BigDecimal(0);
            } else {
                k = y2.subtract(y1).divide(x2.subtract(x1),30, RoundingMode.HALF_UP);
            }
            if (i == 1) {
                lastk = k;
            } else {
                if (!k.equals(lastk)) {
                    ans++;
                    lastk = k;
                }
            }
        }
        return ans;
    }

    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map = new HashMap<>();
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = nums2.length - 1;i >= 0; i--) {
            int num = nums2[i];
            while (!stack.isEmpty() && num > stack.peek()) {
                stack.pop();
            }
            map.put(num, stack.isEmpty() ? -1 : stack.peek());
            stack.push(num);
        }
        int[] ans = new int[nums1.length];
        for (int i = 0;i < nums1.length; i++) {
            ans[i] = map.get(nums1[i]);
        }
        return ans;
    }

    public int totalStrength(int[] strength) {
        final int mod = (int) (1e9 + 7);
        int n = strength.length;
        int[] left = new int[n];
        int[] right = new int[n];
        Arrays.fill(right, n);
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0;i < n; i++) {
            while (!stack.isEmpty() && strength[stack.peek()] >= strength[i]) {
                right[stack.pop()] = i;
            }
            left[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }
        long s = 0;
        long[] ss = new long[n + 2];
        for (int i = 1;i <= n; i++) {
            s += strength[i - 1];
            ss[i + 1] = (ss[i] + s) % mod;
        }
        long ans = 0;
        for (int i = 0;i < n; i++) {
            int l = left[i] + 1, r = right[i] - 1;
            long tot = ((long) (i - l + 1) * (ss[r + 2] - ss[i + 1]) - (long)(r - i + 1) * (ss[i + 1] - ss[l])) % mod;
            ans = (ans + strength[i] * tot) % mod;
        }
        return (int) ((ans + mod) % mod);
    }

    public int cutOffTree(List<List<Integer>> forest) {
        List<int[]> trees = new ArrayList<>();
        for (int i = 0;i < forest.size(); i++) {
            for (int j = 0;j < forest.get(i).size(); j++) {
                if (forest.get(i).get(j) > 1) {
                    trees.add(new int[]{i, j, forest.get(i).get(j)});
                }
            }
        }
        trees.sort(Comparator.comparingInt(o -> o[2]));
        int x = 0, y = 0, ans = 0;
        for (int[] cur : trees) {
            int nx = cur[0], ny = cur[1];
            int d = bfs(x, y, nx, ny, forest);
            if (d == -1) {
                return -1;
            }
            ans += d;
            x = nx;
            y = ny;
        }
        return ans;
    }
    int[][] dirs = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
    public int bfs(int x, int y, int s, int e, List<List<Integer>> forest) {
        if (x == s && y == e) {
            return 0;
        }
        int n = forest.size();
        int m = forest.get(0).size();
        boolean[][] vis = new boolean[n][m];
        Deque<int[]> que = new ArrayDeque<>();
        que.add(new int[]{x,y,0});
        vis[x][y] = true;
        while (!que.isEmpty()) {
            for (int i = que.size() - 1;i >= 0; i--) {
                int[] cur = que.pollFirst();
                int curX = cur[0], curY = cur[1];
                int val = cur[2];
                for (int[] dir : dirs) {
                    int nx = curX + dir[0];
                    int ny = curY + dir[1];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m) {
                        continue;
                    }
                    if (forest.get(nx).get(ny) == 0 || vis[nx][ny]) {
                        continue;
                    }
                    if (nx == s && ny == e) {
                        return val + 1;
                    }
                    que.addLast(new int[]{nx, ny, val + 1});
                    vis[nx][ny] = true;
                }
            }
        }
        return -1;
    }

    public int findSubstringInWraproundString(String p) {
        int[] dp = new int[26];
        int k = 0;
        for (int i = 0;i < p.length(); i++) {
            if (i > 0 &&
                    ((p.charAt(i) - p.charAt(i - 1) == 1) || ((p.charAt(i) - 1 + 26) % 26) + 'a' == p.charAt(i - 1))) {
                k++;
            } else {
                k = 1;
            }
            dp[p.charAt(i) - 'a'] = Math.max(dp[p.charAt(i) - 'a'], k);
        }
        return Arrays.stream(dp).sum();
    }

    public String removeOuterParentheses(String s) {
        int cur = 0;
        int vv = -1;
        StringBuilder sb = new StringBuilder();
        for (int i = 0;i < s.length(); i++) {
            char x = s.charAt(i);
            if (x == '(') {
                if (cur == 0) {
                    vv = i;
                }
                cur++;
            } else {
                cur--;
            }
            if (cur == 0) {
                sb.append(s, vv + 1, i);
            }
        }
        return sb.toString();
    }

    public int findClosest(String[] words, String word1, String word2) {
        int n = words.length;
        int last1 = -1;
        int last2 = -1;
        int min = Integer.MAX_VALUE;
        for (int i = 0;i < n; i++) {
            if (words[i].equals(word1)) {
                if (last2 != -1) {
                    min = Math.min(min, i - last1);
                }
                last1 = i;
            } else if (words[i].equals(word2)) {
                if (last1 != -1) {
                    min = Math.min(min, i - last2);
                }
                last2 = i;
            }
        }
        return min;
    }

    public boolean digitCount(String num) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0;i < num.length(); i++) {
            Integer cur = Integer.parseInt("" + num.charAt(i));
            map.put(cur, map.getOrDefault(cur, 0) + 1);
        }
        for (int i = 0;i < num.length(); i++) {
            int cur = Integer.parseInt("" + num.charAt(i));
            int vv = 0;
            if (map.get(i) != null) {
                vv = map.get(i);
            }
            if (cur != vv) {
                return false;
            }
        }
        return true;
    }

    public String largestWordCount(String[] messages, String[] senders) {
        Map<String, Set<String>> map = new HashMap<>();
        for (int i = 0;i < messages.length; i++) {
            map.putIfAbsent(senders[i], new HashSet<>());
            String[] strings = messages[i].split(" ");
            for (String s : strings) {
                map.get(senders[i]).add(s);
            }
        }
        int max = 0;
        List<String> ans = new ArrayList<>();
        for (String s : map.keySet()) {
            if (map.get(s).size() > max) {
                max = map.get(s).size();
            }
        }
        for (String s : map.keySet()) {
            if (map.get(s).size() == max) {
                ans.add(s);
            }
        }
        ans.sort(Comparator.reverseOrder());
        return ans.get(0);
    }

    public long maximumImportance(int n, int[][] roads) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        Map<Integer, Integer> map1 = new HashMap<>();
        int[][] curArr = new int[n][2];
        for (int[] road : roads) {
            map.put(road[0], map.getOrDefault(road[0], 0) + 1);
            map.put(road[1], map.getOrDefault(road[1], 0) + 1);
        }
        int index = 0;
        for (Integer key : map.keySet()) {
            curArr[index] = new int[]{key, map.get(key)};
        }
        Arrays.sort(curArr, (o1, o2) -> o2[1] -o1[1]);
        int cur = n;
        for (int i = 0;i < curArr.length; i++) {
            if (map1.get(curArr[i][0]) != null) {
                continue;
            }
            map1.put(curArr[i][0], cur--);
        }
        long ans = 0;
        for (int[] road : roads) {
            ans += map1.get(road[0]) + map1.get(road[1]);
        }
        return ans;
    }

    public int rearrangeCharacters(String s, String target) {
        Map<Character, Integer> map = new HashMap<>();
        Map<Character, Integer> map1 = new HashMap<>();
        for (int i = 0;i < s.length(); i++) {
            char c = s.charAt(i);
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        for (int i = 0;i < target.length(); i++) {
            char c = target.charAt(i);
            map1.put(c, map.getOrDefault(c, 0) + 1);
        }
        int ans = Integer.MAX_VALUE;
        for (char x : map1.keySet()) {
            int cur = map1.get(x);
            if (map.get(x) == null) {
                return 0;
            }
            int vv = map.get(x);
            ans = Math.min(ans, vv / cur);
        }
        return ans;
    }

    public String discountPrices(String sentence, int discount) {
        String[] strings = sentence.split(" ");
        for (int i = 0;i < strings.length; i++) {
            String s = strings[i];
            StringBuilder sb = new StringBuilder();
            if (s.charAt(0) == '$') {
                sb.append("$");
                double cur;
                try {
                    cur = Double.parseDouble(s.substring(1));
                } catch (Exception e) {
                    continue;
                }
                cur = cur - cur * (double) (discount) / 100;
                BigDecimal two = new BigDecimal(cur);
                two = two.setScale(2, RoundingMode.HALF_UP);
                sb.append(two);
                strings[i] = sb.toString();
            }
        }
        StringBuilder ans = new StringBuilder();
        for (String s : strings) {
            ans.append(s);
            ans.append(" ");
        }
        return ans.toString().substring(0, ans.length() - 1);
    }

    public int minimumObstacles(int[][] grid) {
        int n = grid.length, m = grid[0].length;
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 0;i < n; i++) {
            Arrays.fill(dp[i], 0x3f3f3f3f);
        }
        Deque<int[]> que = new ArrayDeque<>();
        que.add(new int[]{0, 0});
        dp[0][0] = 0;
        while (!que.isEmpty()) {
            int[] cur = que.pollFirst();
            int x = cur[0], y = cur[0];
            for (int i = 0;i < 4; i++) {
                int nx = x + dirs[i][0];
                int ny = y + dirs[i][1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < m) {
                    if (dp[nx][ny] > dp[x][y] + grid[nx][ny]) {
                        dp[nx][ny] = dp[x][y] + grid[nx][ny];
                        if (grid[nx][ny] == 1) {
                            que.addLast(new int[]{nx, ny});
                        } else {
                            que.addFirst(new int[]{nx, ny});
                        }
                    }
                }
            }
        }
        return dp[n - 1][m - 1];
    }

    public int totalSteps(int[] nums) {
        int ans = 0;
        Deque<int[]> que = new ArrayDeque<>();
        for (int cur : nums) {
            int maxT = 0;
            while (!que.isEmpty() && que.peek()[0] <= cur) {
                maxT = Math.max(maxT, que.peek()[1]);
                que.pop();
            }
            if (!que.isEmpty()) {
                maxT++;
            }
            ans = Math.max(ans, maxT);
            que.push(new int[]{cur, maxT});
        }
        return ans;
    }

    private StringBuilder sb = new StringBuilder();

    public int sumRootToLeaf(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            sb.append(root.val);
            int cur = Integer.parseInt(sb.toString(), 2);
            System.out.println(cur);
            sb.deleteCharAt(sb.length() - 1);
            return cur;
        }
        sb.append(root.val);
        return sumRootToLeaf(root.left) + sumRootToLeaf(root.right);
    }

    public int minMaxGame(int[] nums) {
        int n = nums.length;
        List<Integer> list = new ArrayList<>();
        List<Integer> newList = new ArrayList<>();
        for (int i = 0;i < n; i++) {
            list.add(nums[i]);
        }
        while (list.size() > 1) {
            for (int i = 0;i < list.size() / 2; i++) {
                if (i % 2 == 0) {
                    newList.add(Math.min(list.get(i * 2), list.get(i * 2 + 1)));
                } else {
                    newList.add(Math.max(list.get(i * 2), list.get(i * 2 + 1)));
                }
            }
            list = newList;
            newList.clear();
        }
        return list.get(0);
    }

    public int partitionArray(int[] nums, int k) {
        Arrays.sort(nums);
        int s = 0, e = 0;
        int ans = 0;
        while (s < nums.length) {
            int cur = nums[s] + k;
            while (e < nums.length && nums[e] - nums[s] <= k) {
                e++;
            }
            ans++;
            s = e;
        }
        return ans;
    }

    public int[] arrayChange(int[] nums, int[][] operations) {
        int n = nums.length;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0;i < n; i++) {
            map.put(nums[i], i);
        }
        for (int[] op : operations) {
            int cur = map.get(op[0]);
            map.remove(op[0]);
            map.put(op[1], cur);
        }
        int[] ans = new int[n];
        for (int x : map.keySet()) {
            ans[map.get(x)] = x;
        }
        return ans;
    }

    public int numUniqueEmails(String[] emails) {
        Set<String> set = new HashSet<>();
        for (String s : emails) {
            int cur = s.indexOf('@');
            String s1 = s.substring(0, cur);
            if (s1.indexOf('+') != -1) {
                s1 = s1.substring(0, s1.indexOf('+'));
            }
            StringBuilder sb = new StringBuilder(s1);
            for (int i = sb.length() - 1;i >= 0; i--) {
                if (sb.charAt(i) == '.') {
                    sb.deleteCharAt(i);
                }
            }
            set.add(sb + s.substring(cur));
        }
        return set.size();
    }

    public int minEatingSpeed(int[] piles, int h) {
        Arrays.sort(piles);
        int l = 1, r = (int) 1e9 + 10;
        while (l < r) {
            int mid = (l + r) >> 1;
            if (check(mid, piles, h)) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    private boolean check(int mid, int[] piles, int h) {
        int cur = piles.length - 1;
        for (int x : piles) {
            cur += x % mid == 0 ? x / mid : x / mid + 1;
        }
        return cur <= h;
    }

    public int minFlipsMonoIncr(String s) {
        int[][] dp = new int[100010][2];
        dp[0][0] = 0;
        dp[0][1] = 0;
        for (int i = 1;i <= s.length(); i++) {
            int cur = s.charAt(i - 1) - '0';
            dp[i][0] = dp[i - 1][0] + (cur == 0 ? 0 : 1);
            dp[i][1] = Math.min(dp[i - 1][0], dp[i - 1][1]) + (cur == 0 ? 1 : 0);
        }
        return Math.min(dp[s.length()][0], dp[s.length()][1]);
    }

    public boolean strongPasswordCheckerII(String password) {
        if (password.length() < 8) {
            return false;
        }
        String s = "!@#$%^&*()-+";
        Set<Character> set = new HashSet<>();
        for (int i = 0;i < s.length(); i++){
            set.add(s.charAt(i));
        }

        boolean flag1 = false;
        boolean flag2 = false;
        boolean flag3 = false;
        boolean flag4 = false;
        boolean flag5 = true;
        for (int i = 0;i < password.length(); i++) {
            char x = password.charAt(i);
            if (x >='a' && x <='z') {
                flag1 = true;
            }
            if (x >= 'A' && x <='Z') {
                flag2 = true;
            }
            if (x >='0' && x <= '9') {
                flag3 = true;
            }
            if (set.contains(x)) {
                flag4 = true;
            }
        }
        for (int i = 1;i < password.length(); i++) {
            if (s.charAt(i) != s.charAt(i - 1)) {
                flag5 = false;
            }
        }
        return flag1 && flag2 && flag3 && flag4 && flag5;
    }

    public int[] successfulPairs(int[] spells, int[] potions, long success) {
        Arrays.sort(potions);
        int[] ans = new int[spells.length];
        for (int i = 0;i < spells.length; i++) {
            int l = 0, r = potions.length;
            while (l < r) {
                int mid = (l + r) >> 1;
                if ((long) potions[mid] * spells[i] >= success) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }

            ans[i] = potions.length - l;
        }
        return ans;
    }

    public boolean matchReplacement(String s, String sub, char[][] mappings) {
        int n = s.length();
        int m = sub.length();
        boolean flag = false;
        Map<Character, Set<Character>> map = new HashMap<>();
        for (char[] chars : mappings) {
            map.putIfAbsent(chars[1], new HashSet<>());
            map.get(chars[1]).add(chars[0]);
        }
        for (int i = 0;i <= n - m; i++) {
            String cur = s.substring(i, i + m);
            boolean cc = true;
            for (int j = 0;j < m; j++) {
                char x = cur.charAt(j);
                char y = sub.charAt(j);
                if (x == y) {
                    continue;
                } else {
                    if (map.get(x) != null && map.get(x).contains(y)) {
                        continue;
                    }
                }
                cc = false;
                break;
            }
            if (cc) {
                flag = true;
                break;
            }
        }
        return flag;
    }

    public long countSubarrays(int[] nums, long k) {
        long l = 0, r = 0, cur = 0;
        long count = 0;
        while (r < nums.length) {
            cur += nums[(int) r];
            while ((long) cur * (r - l + 1) >= k){
                cur -= nums[(int) l];
                l++;
            }
            count += (r - l + 1);
            r++;
        }
        return count;
    }

    public double calculateTax(int[][] brackets, int income) {
        double ans = 0;
        int last = 0;
        for (int[] b : brackets) {
            if (income > 0) {
                int cur = Math.min(b[0], income);
                cur -= last;
                ans += cur * b[1] / 100.0;
                income -= b[0];
                last = b[0];
            } else {
                break;
            }
        }
        return ans;
    }

    public int minPathCost(int[][] grid, int[][] moveCost) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] min = new int[m][n];
        for (int i = 1;i < m; i++) {
            Arrays.fill(min[i], Integer.MAX_VALUE);
        }
        for (int v = 1;v < m; v++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int cur = grid[v - 1][j];
                    min[v][i] = Math.min(cur + moveCost[cur][i] + min[v - 1][j], min[v][i]);
                }
            }
        }
        int ans = Integer.MAX_VALUE;
        for (int i = 0;i < n; i++) {
            ans = Math.min(ans, min[m - 1][i] + grid[m - 1][i]);
        }
        return ans;
    }

    public int distributeCookies(int[] cookies, int k) {
        int n = cookies.length;
        int[] sum = new int[(1 << n)];
        for (int i = 0;i < (1 << n); i++) {
            for (int j = 0;j < n; j++) {
                if ((i & (1 << j)) == 1) {
                    sum[i] += cookies[j];
                }
            }
        }
        int[][] dp = new int[k][(1 << n)];
        dp[0] = sum;
        for (int i = 1;i < k; i++) {
            for (int j = 0;j < (1 << n); j++) {
                dp[i][j] = (int) 1e9;
                for (int s = j; s > 0; s = (s - 1) & j) {
                    dp[i][j] = Math.min(dp[i][j], Math.max(dp[i - 1][j ^ s], sum[s]));
                }
            }
        }
        return dp[k - 1][(1 << n) - 1];
    }

    public long distinctNames(String[] ideas) {
        Set<String>[] sets = new Set[26];
        for (int i = 0;i < 26; i++) {
            sets[i] = new HashSet<>();
        }
        for (String s : ideas) {
            sets[s.charAt(0) - '0'].add(s.substring(1));
        }
        long ans = 0;
        for (int i = 0;i < 26; i++) {
            for (int j = 0;j < i; j++) {
                int m = 0;
                for (String s : sets[i]) {
                    if (sets[j].contains(s)) {
                        m++;
                    }
                }
                ans += (long) (sets[i].size() - m) * (sets[j].size() - m);
            }
        }
        return ans * 2;
    }

    public int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length;
        int l = 0, r = nums[n - 1] - nums[0];
        while (l <= r) {
            int mid = (l + r) >> 1;
            int cnt = 0;
            for (int i = 0, j = 0;j < n; j++) {
                while (nums[j] - nums[i] > mid) {
                    i++;
                }
                cnt += j - i;
            }
            if (cnt >= k) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    public int binarySearch(int[] nums, int end, int cur) {
        int l = 0, r = end;
        while (l < r) {
            int mid = (l + r) >> 1;
            if (nums[mid] < cur) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return l;
    }

    public int findPairs(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        int ans = 0;
        for (int x : map.keySet()) {
            if (k != 0) {
                if (map.get(x + k) != null) {
                    ans++;
                }
            } else {
                if (map.get(x) > 1) {
                    ans++;
                }
            }
        }
        return ans;
    }

    public int[] findFrequentTreeSum(TreeNode root) {
        Map<Integer, Integer> map = new HashMap<>();
        dfs(root, map);
        int max = 0;
        for (int x : map.keySet()) {
            max = Math.max(max, map.get(x));
        }
        int cur = 0;
        for (int x : map.keySet()) {
            if (map.get(x) == max) {
                cur++;
            }
        }
        int[] ans = new int[cur];
        cur = 0;
        for (int x : map.keySet()) {
            if (map.get(x) == max) {
                ans[cur++] = x;
            }
        }
        return ans;
    }

    public int dfs(TreeNode root, Map<Integer, Integer> map) {
        if (root == null) {
            return 0;
        }
        int cur = root.val + dfs(root.left, map) + dfs(root.right, map);
        map.put(cur, map.getOrDefault(cur, 0) + 1);
        return cur;
    }

    public String greatestLetter(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0;i < s.length(); i++) {
            map.put(s.charAt(i), 1);
        }
        char x = 0;
        for (char v : map.keySet()) {
            if (v >= 'A' && v <= 'Z') {
                char vv = Character.toLowerCase(v);
                if (map.get(vv) != null) {
                    if (v > x) {
                        x = v;
                    }
                }
            }
        }
        if (x == 0) {
            return "";
        } else {
            return String.valueOf(x);
        }
    }


    public int minimumNumbers(int num, int k) {
        for (int i = 1;;i++) {
            int cur = i * k;
            if ((num - cur) % 10 == 0){
                return i;
            }
            if (cur > num) {
                return -1;
            }
        }
    }

    public int longestSubsequence(String s, int k) {
        int cur = 0;
        int vv = k;
        while (vv > 0) {
            cur++;
            vv >>= 1;
        }
        int c1 = 0, c2 = 0;
        StringBuilder sb = new StringBuilder();
        boolean flag = false;
        for (int i = s.length() - 1;i >= 0; i--) {
            char x = s.charAt(i);
            if (x == '1') {
                flag = true;
            }
            if (c1 + c2 < cur) {
                if (x == '0') {
                    c1++;
                    sb.insert(0, "0");
                } else {
                    sb.insert(0, "1");
                    if (Long.parseLong(sb.toString(), 2) > k) {
                        sb.deleteCharAt(sb.length() - 1);
                        continue;
                    }
                    c2++;
                }
            } else {
                if (x == '0') {
                    sb.insert(0, "0");
                    c1++;
                }
            }
        }
        if (!flag) {
            return s.length();
        }
        return c1 + c2;
    }

    public long sellingWood(int m, int n, int[][] prices) {
        int[][] pr = new int[m + 1][n + 1];
        for (int[] p : prices) {
            pr[p[0]][p[1]] = p[2];
        }
        long[][] dp = new long[m + 1][n + 1];
        for (int i = 1;i <= m; i++) {
            for (int j = 1;j <= n; j++) {
                dp[i][j] = pr[i][j];
                for (int k = 1;k < i; k++) {
                    dp[i][j] = Math.max(dp[i][j], dp[k][j] + dp[i - k][j]);
                }
                for (int k = 1;k < j; k++) {
                    dp[i][j] = Math.max(dp[i][j], dp[i][k] + dp[i][j - k]);
                }
            }
        }
        return dp[m][n];
    }

    private int ans0625;
    private int cur0625;

    public int findBottomLeftValue(TreeNode root) {
        cur0625 = 0;
        findLeft(root.left, true, 1);
        findLeft(root.right, false, 1);
        return ans0625;
    }

    private void findLeft(TreeNode root, boolean b, int cur) {
        if (root == null) {
            return ;
        }
        if (b && cur > cur0625) {
            cur0625 = cur;
            ans0625 = root.val;
        }
        findLeft(root.left, true, 1);
        findLeft(root.right, false, 1);
    }

    public List<Integer> largestValues(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        Queue<TreeNode> que = new ArrayDeque<>();
        que.add(root);
        while (!que.isEmpty()) {
            int max = Integer.MIN_VALUE;
            for (int i = que.size() - 1;i >= 0; i--) {
                TreeNode node = que.poll();
                max = Math.max(max, node.val);
                if (node.left != null) {
                    que.add(node.left);
                }
                if (node.right != null) {
                    que.add(node.right);
                }
            }
            ans.add(max);
        }
        return ans;
    }

    public int minCost(int[][] costs) {
        int[] dp = new int[3];
        int n = costs.length;
        for (int i = 0;i < 3; i++) {
            dp[i] = costs[0][i];
        }
        for (int i = 1;i < n; i++) {
            int[] dpNew = new int[3];
            for (int j = 0;j < 3; j++) {
                dpNew[j] = Math.min(dp[(j + 1) % 3], dp[(j + 2) % 3]) + costs[i][j];
            }
            dp = dpNew;
        }
        return Math.min(dp[0], Math.min(dp[1], dp[2]));
    }

    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> ans = new ArrayList<>();
        Map<String, Integer> map = new HashMap<>();
        for (String word : words) {
            map.put(word, map.getOrDefault(word, 0) + 1);
        }
        int one = words[0].length(), total = words.length * one;
        for (int i = 0;i < s.length() - total; i++) {
            String curS = s.substring(i, i + total);
            Map<String, Integer> cur = new HashMap<>();
            for (int j = 0;j < total; j += one) {
                String item = curS.substring(j, j + one);
                if (!map.containsKey(item)) {
                    break;
                } else {
                    cur.put(item, map.getOrDefault(item, 0) + 1);
                }
            }
            if (cur.equals(map)) {
                ans.add(i);
            }
        }
        return ans;
    }

    public int countAsterisks(String s) {
        int cur = 0;
        int ans = 0;
        for (int i = 0;i < s.length(); i++) {
            char x = s.charAt(i);
            if (x == '|') {
                cur++;
            }
            if ((cur % 2 == 0) && (x == '*')) {
                ans++;
            }
        }
        return ans;
    }

    public long countPairs(int n, int[][] edges) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] edge : edges) {
            map.putIfAbsent(edge[0], new ArrayList<>());
            map.putIfAbsent(edge[1], new ArrayList<>());
            map.get(edge[0]).add(edge[1]);
            map.get(edge[1]).add(edge[0]);
        }
        int[] vis = new int[n];
        List<Integer> list = new ArrayList<>();
        long ans = 1;
        for (int i = 0;i < n; i++) {
            if (vis[i] == 0) {
                vis[i] = 1;
                int vv = 0;
                Queue<Integer> que = new ArrayDeque<>();
                que.add(i);
                while (!que.isEmpty()) {
                    int cur = que.poll();
                    vv++;
                    if (map.get(cur) == null) {
                        continue;
                    }
                    for (int x : map.get(cur)) {
                        if (vis[x] == 0) {
                            que.add(x);
                            vis[x] = 1;
                        }
                    }
                }
                list.add(vv);
            }
        }
        if (list.size() == 1) {
            return 0;
        }
        long[] sum = new long[list.size() + 1];
        for (int i = 0;i < list.size(); i++) {
            sum[i + 1] = sum[i] + list.get(i);
        }
        for (int i = 0;i < list.size(); i++) {
            ans += list.get(i) * (sum[list.size()] - sum[i + 1]);
        }
        return ans;
    }

    public int maximumXOR(int[] nums) {
        int[] cur = new int[32];
        for (int i = 0;i < nums.length; i++) {
            int vv = 0;
            int v = nums[i];
            while (v > 0) {
                if (v % 2 == 1) {
                    cur[vv] = 1;
                }
                vv++;
                v >>= 1;
            }
        }
        int ans = 0;
        int vv = 1;
        for (int i = 0;i < cur.length; i++) {
            if (cur[i] == 1) {
                ans += vv;
            }
            vv *= 2;
        }
        return ans;
    }

    public int distinctSequences(int n) {
        if (n == 1) {
            return 6;
        }
        if (n == 2) {
            return 22;
        }
        int mod = (int) (1e9 + 7);
        long[][][] dp = new long[n + 1][7][7];
        for (int i = 1;i <= 6; i++) {
            for (int j = 1; j <= 6; j++) {
                for (int k = 1; k <= 6; k++) {
                    if (i != j && i != k && j != k && gcd(i, j) == 1 && gcd(j, k) == 1)
                        dp[3][i][j]++;
                }
            }
        }

        for (int i = 4;i <= n; i++) {
            for (int j = 1;j <= 6; j++) {
                for (int k = 1;k <= 6; k++) {
                    for (int l = 1;l <= 6; l++) {
                        if (j != k && j != l && k != l && gcd(i, j) == 1 && gcd(j, k) == 1) {
                            dp[i][j][k] += dp[i - 1][k][l];
                            dp[i][j][k] %= mod;
                        }
                    }
                }
            }
        }
        long ans = 0;
        for (int i = 1;i <= 6; i++) {
            for (int j = 1;j <= 6; j++) {
                if (i != j) {
                    ans = (ans + dp[n][i][j]) % mod;
                }
            }
        }
        return (int) ans;
    }

    public long gcd(long a, long b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    public boolean checkXMatrix(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        boolean ans = true;
        for (int i = 0;i < m; i++) {
            for (int j = 0;j < n; j++) {
                if (i == j && grid[i][j] == 0) {
                    ans = false;
                    break;
                } else if (j == (n - i - 1) && grid[i][j] == 0) {
                    ans = false;
                    break;
                } else if ((i!=j) && grid[i][j] != 0) {
                    ans = false;
                    break;
                }
            }
            if (!ans) {
                break;
            }
        }
        return ans;
    }

    public int countHousePlacements(int n) {
        int mod = (int) (1e9+7);
        if (n == 1) {
            return 4;
        }
        if (n == 2) {
            return 9;
        }
        long[] dp = new long[n + 1];
        dp[1] = 2;
        dp[2] = 3;
        for (int i = 3;i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
            dp[i] %= mod;
        }
        return (int) (dp[n] * dp[n] % mod);
    }

    public int maximumsSplicedArray(int[] nums1, int[] nums2) {
        List<Integer> list = new ArrayList<>();
        long sum = 0;
        for (int i = 0;i < nums1.length; i++) {
            list.add(nums2[i] - nums1[i]);
            sum += nums1[i];
        }
        long ans = sum + maxSum(list);

        list.clear();
        sum = 0;
        for (int i = 0;i < nums2.length; i++) {
            list.add(nums1[i] - nums2[i]);
            sum += nums2[i];
        }
        return (int) Math.max(ans, sum + maxSum(list));
    }

    public int maxSum(List<Integer> list) {
        int cur = 0;
        int max = 0;
        for (int x : list) {
            cur += x;
            if (cur < 0) {
                cur = 0;
            }
            max = Math.max(max, cur);
        }
        return max;
    }

    private List<Integer>[] g;
    int[] nums, xor, in, out;
    int clock;

    public int minimumScore(int[] nums, int[][] edges) {
        int n = nums.length;
        g = new ArrayList[n];
        for (int i = 0;i < n; i++) {
            g[i] = new ArrayList<>();
        }
        this.nums = nums;
        this.xor = new int[n];
        this.in = new int[n];
        this.out = new int[n];

        for (int[] edge : edges) {
            int x = edge[0];
            int y = edge[1];
            g[x].add(y);
            g[y].add(x);
        }
        dfs(0, -1);
        int ans = Integer.MAX_VALUE;
        for (int i = 2;i < n; i++) {
            for (int j = 1;j < i; j++) {
                int x, y, z;
                if (isAncestor(i, j)) {
                    x = xor[i] ^ xor[j];
                    y = xor[j];
                    z = xor[0] ^ xor[i];
                } else if (isAncestor(j, i)) {
                    x = xor[i];
                    y = xor[j] ^ x;
                    z = xor[0] ^ xor[j];
                } else {
                    x = xor[i];
                    y = xor[j];
                    z = xor[0] ^ xor[i] ^ xor[j];
                }
                ans = Math.min(ans, Math.max(x, Math.max(y, z)) - Math.min(x, Math.min(y, z)));
                if (ans == 0) {
                    return 0;
                }
            }
        }
        return ans;
    }

    public void dfs(int x, int father) {
        in[x] = clock++;
        xor[x] = nums[x];
        for (int vv : g[x]) {
            if (vv != father) {
                dfs(vv, x);
                xor[x] ^= xor[vv];
            }
        }
        out[x] = clock;
    }

    public int findLUSlength(String[] strs) {
        int n = strs.length;
        int ans = -1;
        for (int i = 0;i < n; i++) {
            boolean check = true;
            for (int j = 0;j < n; j++) {
                if (i != j && isSubque(strs[i], strs[j])) {
                    check = false;
                    break;
                }
            }
            if (check) {
                ans = Math.max(ans, strs[i].length());
            }
        }
        return ans;
    }

    private boolean isSubque(String str, String str1) {
        int i = 0;
        for (int j = 0;j < str1.length(); j++) {
            if (str.charAt(i) == str.charAt(j)) {
                i++;
            }
        }
        return i == str.length();
    }

    public boolean isAncestor(int x, int y) {
        return in[x] < in[y] && in[y] <= out[x];
    }

    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        int n = stations.length;
        // 到达第i站，加了j次油最大次数
        long[][] dp = new long[n + 1][n + 1];
        for (int i = 0;i < n; i++) {
            dp[i][0] = startFuel;
        }
        // 枚举
        for (int i = 1;i <= n; i++) {
            if (dp[i - 1][i - 1] < stations[i - 1][0]) {
                return -1;
            }
            for (int j = 1;j <= i; j++) {
                if (dp[i - 1][j - 1] >= stations[i - 1][0]) {
                    dp[i][j] = dp[i - 1][j - 1] + stations[i - 1][0];
                }
                if (i > j) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][j]);
                }
            }
        }
        for (int i = 0;i <= stations.length; i++) {
            if (target <= dp[n][i]) {
                return i;
            }
        }
        return -1;
    }

    public int deleteAndEarn(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        } else if (nums.length == 1) {
            return nums[0];
        }
        int len = nums.length;
        int max = nums[0];
        for (int i = 0;i < len; i++) {
            max = Math.max(nums[i], max);
        }
        int[] all = new int[max + 1];
        for (int x : nums) {
            all[x]++;
        }
        int[] dp = new int[max + 1];
        dp[1] = all[1];
        dp[2] = Math.max(dp[1], all[2] * 2);
        for (int i = 3;i <= max; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + all[i] * i);
        }
        return dp[max];
    }

    public String decodeMessage(String key, String message) {
        int index = 0;
        Map<Character, Character> map = new HashMap<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0;i < key.length(); i++) {
            char x = key.charAt(i);
            if (x != ' ' && !map.containsKey(x)) {
                map.put(x, (char) (index + 'a'));
                index++;
            }
        }
        for (int i = 0;i < message.length(); i++) {
            char x = message.charAt(i);
            if (x == ' ') {
                sb.append(x);
            } else {
                sb.append(map.get(x));
            }
        }
        return sb.toString();
    }

    public int[][] spiralMatrix(int m, int n, ListNode head) {
        int[][] mar = new int[m][n];
        for (int i = 0;i < m; i++) {
            Arrays.fill(mar[i], -1);
        }
        int left = 0;
        int right = n - 1;
        int top = 0;
        int bottom = m - 1;
        while (head != null) {
            for (int i = left;i <= right && head != null; i++) {
                mar[top][i] = head.val;
                head = head.next;
            }
            top++;
            for (int i = top;i <= bottom && head != null; i++) {
                mar[i][right] = head.val;
                head = head.next;
            }
            right--;
            for (int i = right;i >= left && head != null; i--) {
                mar[bottom][i] = head.val;
                head = head.next;
            }
            bottom--;
            for (int i = bottom;i >= top && head != null; i--) {
                mar[i][left] = head.val;
                head = head.next;
            }
            left++;
        }
        return mar;
    }

    public int peopleAwareOfSecret(int n, int delay, int forget) {
        int mod = (int) (1e9 + 7);
        // 当前知道秘密的人数
        long ans = 0;
        long[] delayArr = new long[n * 2];
        long[] forgetArr = new long[n * 2];
        delayArr[1] = 1;
        for (int i = 1;i <= n; i++) {
            long curNew = delayArr[i];
            for (int j = i + delay;j < i + forget; j++) {
                delayArr[j] += curNew;
                delayArr[j] %= mod;
            }
            forgetArr[i + forget] = curNew * -1;
            forgetArr[i + forget] %= mod;
            ans += curNew;
            ans += forgetArr[i];
            ans %= mod;
        }
        return (int) ans;
    }

    private int mod = (int) (1e9+7);

    public int countPaths(int[][] grid) {
        int n = grid.length;
        int m = grid[0].length;
        int[][] dp = new int[n][m];
        int ans = 0;
        for (int i = 0;i < n; i++) {
            for (int j = 0;j < m; j++) {
                ans += dp(i, j, dp, n, m, grid);
                ans %= mod;
            }
        }
        return ans;
    }

    public int dp(int x, int y, int[][] dp, int n, int m, int[][] grid) {
        if (dp[x][y] > 0) {
            return dp[x][y];
        }
        dp[x][y] = 1;
        for (int[] dir : dirs) {
            int nx = x + dir[0];
            int ny = y + dir[1];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m && grid[nx][ny] > grid[x][y]) {
                dp[x][y] += dp(nx, ny, dp, n, m, grid);
                dp[x][y] %= mod;
            }
        }
        return dp[x][y];
    }

    public boolean canJump(int[] nums) {
        int right = 0;
        for (int i = 0;i < nums.length; i++) {
            if (i <= right) {
                right = Math.max(right, i + nums[i]);
            }
            if (right >= n - 1){
                return true;
            }
        }
        return false;
    }

    public int maxSubArray(int[] nums) {
        int max = Integer.MIN_VALUE;
        int cur = 0;
        for (int i = 0;i < nums.length; i++) {
            cur += nums[i];
            if (cur < 0) {
                cur = 0;
            }
            max = Math.max(cur, max);
        }
        return max;
    }

    public int maxProduct(int[] nums) {
        int n = nums.length;
        int[] max = new int[n];
        int[] min = new int[n];
        max[0] = nums[0];
        min[0] = nums[0];
        for (int i = 1;i < nums.length; i++) {
            max[i] = Math.max(max[i - 1] * nums[i], Math.max(min[i - 1] * nums[i], nums[i]));
            min[i] = Math.min(min[i - 1] * nums[i], Math.min(min[i - 1] * nums[i], nums[i]));
        }
        int asn = Integer.MIN_VALUE;
        for (int i = 0;i < nums.length;i++) {
            asn = Math.min(asn, max[i]);
        }
        return asn;
    }

    public int getMaxLen(int[] nums) {
        int len = nums.length;
        int positive = nums[0] > 0 ? 1 : 0;
        int negative = nums[0] < 0 ? 1 : 0;
        int maxLen = positive;
        for (int i = 1;i < len; i++) {
            if (nums[i] > 0) {
                positive++;
                negative = negative > 0 ? negative + 1 : 0;
            } else if (nums[i] < 0) {
                int newPositive = negative > 0 ? negative + 1 : 0;
                int newNegative = positive + 1;
                positive = newPositive;
                negative = newNegative;
            } else {
                positive = 0;
                negative = 0;
            }
            maxLen = Math.max(maxLen, positive);
        }
        return maxLen;
    }

    /**
     * @Description: 找到第一个波谷，变成波峰并将前面的元素变成倒序
     * @return:
     */
    public int nextGreaterElement(int n) {
        List<Integer> list = new ArrayList<>();
        while (n != 0) {
            list.add(n % 10);
            n /= 10;
        }
        int size = list.size(), idx = -1;
        for (int i = 0;i < size - 1 && idx == -1; i++) {
            if (list.get(i + 1) < list.get(i)) {
                idx = i + 1;
            }
        }
        if (idx == -1) {
            return -1;
        }
        for (int i = 0;i < idx; i++) {
            if (list.get(i) > list.get(idx)) {
                swapList(list, i, idx);
                break;
            }
        }
        for (int l = 0, r = idx - 1;l != r; l++, r--) {
            swapList(list, l, r);
        }
        long ans = 0;
        for (int i = list.size() - 1;i >= 0; i--) {
            ans = ans * 10 + list.get(i);
        }
        return ans > Integer.MAX_VALUE ? -1 : (int) ans;
    }

    public void swapList(List<Integer> list, int a, int b) {
        int c = list.get(a);
        list.set(a, list.get(b));
        list.set(b, c);
    }

    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1;i < n; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[n - 1][0];
    }

    public boolean evaluateTree(TreeNode root) {
        int ans = back(root);
        return ans == 1;
    }

    public int back(TreeNode root) {
        if (root.left == null && root.right == null) {
            return root.val;
        }
        int left = back(root.left);
        int right = back(root.right);
        if (root.val == 2) {
            return left | right;
        } else {
            return left & right;
        }
    }

    public int latestTimeCatchTheBus(int[] buses, int[] passengers, int capacity) {
        Arrays.sort(buses);
        Arrays.sort(passengers);
        int j = 0,c = 0;
        for (int t : buses) {
            for (c = capacity;c > 0 && passengers[j] <= t; j++) {
                c--;
            }
        }
        j--;
        int ans = c > 0 ? buses[buses.length - 1] : passengers[j];
        while (j >= 0 && passengers[j--] == ans) {
            ans--;
        }
        return ans;
    }

    public long minSumSquareDiff(int[] nums1, int[] nums2, int k1, int k2) {
        int n = nums1.length;
        long k = k1 + k2;
        List<Integer> list = new ArrayList<>();
        list.add(0);
        for (int i = 0;i < n; i++) {
            list.add(Math.abs(nums1[i] - nums2[i]));
        }
        list.sort(Comparator.comparingInt(o -> o));
        for (int i = n;i > 0; i--) {
            int x = (int) (k / (n - i + 1));
            if (x >= (list.get(i) - list.get(i - 1))) {
                k -= (long) (n - i + 1) * (list.get(i) - list.get(i - 1));
            } else {
                int t = list.get(i) - x;
                for (int j = i;j <= n; j++) {
                    list.set(j, t);
                }
                int y = (int) (k % (n - i + 1));
                for (int j = 0;j < y; j++) {
                    list.set(i + j, list.get(i + j) - 1);
                }
                long ans = 0;
                for (int j = 1;j <= n; j++) {
                    ans += (long) list.get(j) * list.get(j);
                }
                return ans;
            }
        }
        return 0;
    }

    public int validSubarraySize(int[] nums, int threshold) {
        int n = nums.length;
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(-1);
        for (int i = 0;i < n; i++) {
            while (stack.size() > 1 && nums[stack.peek()] > nums[i]) {
                int cur = nums[stack.pop()];
                int len = i - stack.peek() - 1;
                if (cur > threshold / len) {
                    return len;
                }
            }
            stack.push(i);
        }
        while (stack.size() > 1) {
            int cur = nums[stack.pop()];
            int len = n - stack.peek() - 1;
            if (cur > threshold / len) {
                return len;
            }
        }
        return 0;
    }

    public int fillCups(int[] amount) {
        int ans = 0;
        while (true) {
            Arrays.sort(amount);
            if (amount[1] >= 1) {
                ans++;
                amount[1]--;
                amount[2]--;
            } else if (amount[2] >= 1) {
                ans++;
                amount[2]--;
            }
            if (amount[2] == 0 && amount[1] == 0 && amount[0] == 0) {
                break;
            }
        }
        return ans;
    }

    public boolean canChange(String start, String target) {
        int status1 = 0;
        int status2 = 0;
        if (start.equals(target)) {
            return true;
        }
        for (int i = 0;i < start.length(); i++) {
            char x = start.charAt(i);
            char y = target.charAt(i);
            if (status1 == 0 && status2 == 0) {
                if (x == y) {
                    continue;
                }
            }
            if (status1 == 0 && x == 'L') {
                return false;
            }
            if (status2 == 0 && y == 'R') {
                return false;
            }
            if (status1 != 0 && status2 != 0) {
                return false;
            }
            if (status1 > 0 && x == 'R') {
                return false;
            } else if (status1 > 0 && x == 'L') {
                status1--;
            }
            if (status2 > 0 && y == 'L') {
                return false;
            } else if (status2 > 0 && y == 'R'){
                status2--;
            }
            if (y == 'L') {
                status1++;
            }
            if (x == 'R') {
                status2++;
            }
        }
        if (status1 != 0 || status2 != 0) {
            return false;
        }
        return true;
    }

    public int lenLongestFibSubseq(int[] arr) {
        int n = arr.length;
        int ans = 0;
        int[][] dp = new int[n][n];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0;i < n; i++) {
            map.put(arr[i], i);
        }
        for (int i = 0;i < n; i++) {
            for (int j = i - 1;j >= 0 && arr[j] * 2 < i; j--) {
                if (arr[i] - arr[j] >= arr[j]) {
                    break;
                }
                int cur = map.getOrDefault(arr[i] - arr[j], -1);
                if (cur >= 0) {
                    dp[i][j] = Math.max(3, dp[j][cur] + 1);
                }
                ans = Math.max(dp[i][j], ans);
            }
        }
        return ans;
    }

    public int[] numberOfPairs(int[] nums) {
        int cur = 0;
        int n = nums.length;
        List<Integer> list = new ArrayList<>();
        for (int i = 0;i < n; i++) {
            list.add(nums[i]);
        }
        while (true) {
            boolean flag = false;
            int index1 = 0;
            int index2 = 0;
            for (int i = 0;i < list.size(); i++) {
                for (int j = i + 1;j < list.size(); j++) {
                    if (list.get(i) == list.get(j)) {
                        index1 = i;
                        index2 = j;
                        flag = true;
                        break;
                    }
                }
            }
            if (!flag) {
                break;
            } else {
                cur++;
                list.remove(index2);
                list.remove(index1);
            }
        }
        return new int[]{cur, list.size()};
    }

    public int maximumSum(int[] nums) {
        int n = nums.length;
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0;i < n; i++) {
            int cur = nums[i];
            int vv = 0;
            while (cur > 0) {
                vv += cur % 10;
                cur /= 10;
            }
            map.putIfAbsent(vv, new ArrayList<>());
            map.get(vv).add(nums[i]);
        }
        int ans = -1;
        for (int k : map.keySet()) {
            List<Integer> cur = map.get(k);
            if (cur.size() >= 2) {
                Collections.sort(cur, (o1, o2) -> o2 - o1);
                ans = Math.max(ans, cur.get(0) + cur.get(1));
            }
        }
        return ans;
    }

    public int[] smallestTrimmedNumbers(String[] nums, int[][] queries) {
        Map<Integer, List<PP>> map = new HashMap<>();
        int n = nums[0].length();
        for (int i = 1;i <= nums[0].length(); i++) {
            List<PP> vv = new ArrayList<>();
            for (int j = 0;j < nums.length; j++) {
                String cur = nums[j].substring(n - i);
                vv.add(new PP(cur, j));
            }
            Collections.sort(vv, Comparator.comparing(o -> o.v1));
            map.put(i, vv);
        }
        int[] ans = new int[queries.length];
        for (int i = 0;i < queries.length; i++) {
            int x = queries[i][0];
            int t = queries[i][1];
            PP pp = map.get(t).get(x - 1);
            ans[i] = pp.v2;
        }
        return ans;
    }

    public int minOperations(int[] nums, int[] numsDivide) {
        int vv = numsDivide[0];
        int ans = 0;
        for (int i = 1;i < numsDivide.length; i++) {
            if (i == 1) {
                vv = (int) gcd(numsDivide[i], numsDivide[i - 1]);
            } else {
                vv = (int) gcd(vv, numsDivide[i]);
            }
        }
        Set<Integer> set = new HashSet<>();
        for (int i = 0;i < nums.length; i++) {
            set.add(nums[i]);
        }
        Arrays.sort(nums);
        int index = 1;
        boolean flag = false;
        List<Integer> list = new ArrayList<>();
        for (int i = 1;i <= Math.sqrt(vv); i++) {
            if (vv % i == 0) {
                list.add(i);
                if (vv / i != i) {
                    list.add(vv / i);
                }
            }
        }
        Collections.sort(list);
        for (int i = 0;i < list.size(); i++) {
            if (set.contains(i)) {
                index = i;
                flag = true;
                break;
            }
        }
        for (int i = 0;i < nums.length; i++) {
            if (nums[i] < index) {
                ans++;
            } else {
                break;
            }
        }
        if (!flag) {
            return -1;
        }
        return ans;
    }

    public int cherryPickup(int[][] grid) {
        int n = grid.length;
        int[][][] f = new int[n * 2 - 1][n][n];
        for (int i = 0;i < n * 2 - 1; i++) {
            for (int j = 0;j < n; j++) {
                 Arrays.fill(f[i][j], Integer.MIN_VALUE);
            }
        }
        f[0][0][0] = grid[0][0];
        for (int k = 1;k <= 2 * n - 2; k++) {
            for (int x1 = Math.max(k - n + 1, 0);x1 <= Math.min(k, n); x1++) {
                int y1 = k - x1;
                if (grid[x1][y1] == -1) {
                    continue;
                }
                for (int x2 = x1;x2 <= Math.min(k, n); x2++) {
                    int y2 = k - x2;
                    if (grid[x2][y2] == -1) {
                        continue;
                    }
                    int cur = f[k - 1][x1][x2];
                    if (x1 > 0) {
                        cur = Math.max(cur, f[k - 1][x1 - 1][x2]);
                    }
                    if (x2 > 0) {
                        cur = Math.max(cur, f[k - 1][x1][x2 - 1]);
                    }
                    if (x1 > 0 && x2 > 0) {
                        cur = Math.max(cur, f[k - 1][x1 - 1][x2 - 1]);
                    }
                    cur += grid[x1][y1];
                    if (x1 != x2) {
                        cur += grid[x2][y2];
                    }
                    f[k][x1][x2] = cur;
                }
            }
        }
        return Math.max(f[2 * n - 2][n - 1][n - 1], 0);
    }

    public int[] asteroidCollision(int[] asteroids) {
        Deque<Integer> stack = new ArrayDeque<>();
        for (int x : asteroids) {
            boolean ok = true;
            while (!stack.isEmpty() && x < 0 && stack.peek() > 0) {
                int cur1 = stack.peek();
                int cur2 = -x;
                if (cur1 <= cur2) {
                    stack.pop();
                }
                ok = cur1 < cur2;
            }
            if (ok) {
                stack.push(x);
            }
        }
        int[] ans = new int[stack.size()];
        for (int i = stack.size() - 1;i >= 0; i++) {
            ans[i] = stack.pop();
        }
        return ans;
    }

    public int arrayNesting(int[] nums) {
        int ans = 0;
        for (int i = 0;i < nums.length; i++) {
            if (nums[i] != -1) {
                int cur = i;
                int cnt = 0;
                while (nums[cur] != -1) {
                    cnt++;
                    int t = nums[cur];
                    nums[cur] = -1;
                    cur = t;
                }
                ans = Math.max(ans, cnt);
            }
        }
        return ans;
    }

//    public boolean sequenceReconstruction(int[] nums, int[][] sequences) {
//        int n = nums.length;
//        boolean ans = false;
//        int[] in = new int[n + 1];
//        for (int[] sequence : sequences) {
//            in[sequence[1]]++;
//        }
//
//        return ans;
//    }

    public String bestHand(int[] ranks, char[] suits) {
        Map<Character, Integer> map = new HashMap<>();
        Map<Integer, Integer> map1 = new HashMap<>();
        for (int i = 0;i < suits.length; i++) {
            map.put(suits[i], map.getOrDefault(suits[i], 0) + 1);
        }
        for (int x : ranks) {
            map1.put(x, map1.getOrDefault(x, 0) + 1);
        }
        for (char x : map.keySet()) {
            if (map.get(x) == 5) {
                return "Flush";
            }
        }
        for (int x : map1.keySet()) {
            if (map.get(x) >= 3) {
                return "Three of a Kind";
            }
        }
        for (int x : map1.keySet()) {
            if (map.get(x) >= 2) {
                return "Pair";
            }
        }
        return "High Card";
    }

    public long zeroFilledSubarray(int[] nums) {
        long ans = 0;
        for (int i = 0;i < nums.length;) {
            if (nums[i] == 0) {
                int cur = i + 1;
                while (cur < nums.length && nums[cur] == 0) {
                    cur++;
                }
                int len = cur - i;
                ans += (long) len * (1 + len) / 2L;
                i = cur;
            } else {
                i++;
            }
        }
        return ans;
    }

    public int shortestSequence(int[] rolls, int k) {
        int ans = 0;
        Set<Integer> set = new HashSet<>();
        for (int i = 0;i < rolls.length; i++) {
            set.add(rolls[i]);
            if (set.size() == k) {
                ans++;
                set.clear();
            }
        }
        return ans + 1;
    }

    public char repeatedCharacter(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0;i < s.length(); i++) {
            char x = s.charAt(i);
            if (map.get(x) != null) {
                return x;
            }
            map.put(x, 1);
        }
        return 'a';
    }

    public int equalPairs(int[][] grid) {
        int ans = 0;
        int n = grid.length;
        for (int i = 0;i < n; i++) {

            for (int j = 0;j < n; j++) {
                boolean flag = true;
                for (int k = 0;k < n; k++) {
                    if (grid[i][k] != grid[k][j]) {
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    ans++;
                }
            }

        }
        return ans;
    }

    public long countExcellentPairs(int[] nums, int k) {
        int[] count = new int[40];
        long ans = 0;
        Set<Integer> set = new HashSet<>();
        for (int x : nums) {
            count[Integer.bitCount(x)] += set.add(x) ? 1 : 0;
        }
        for (int i = 1;i <= 30; i++) {
            for (int j = Math.min(1, (k - 1));j < 30; j++) {
                ans += (long) count[i] * count[j];
            }
        }
        return ans;
    }

    public TreeNode pruneTree(TreeNode root) {
        if (root == null) {
            return root;
        }
        root.left = pruneTree(root.left);
        root.right = pruneTree(root.right);
        if (root.left == null && root.right == null && root.val == 0) {
            return null;
        }
        return root;
    }

    public String fractionAddition(String expression) {
        List<int[]> list = new ArrayList<>();
        StringBuilder sb = new StringBuilder(expression);
        if (sb.charAt(0) != '-') {
            sb.insert(0, '+');
        }
        int index = 0;
        int m = -1;
        while (index < sb.length()) {
            char cur = sb.charAt(index);
            if (cur == '+' || cur == '-') {
                int next = index + 1;
                while (next < sb.length()
                        && (sb.charAt(next) != '+' || sb.charAt(next) != '-')) {
                    next++;
                }
                String curs = sb.substring(index + 1, next);
                String[] strings = curs.split("/");
                int flag = sb.charAt(index) == '+' ? 1 : -1;
                int cur1 = Integer.parseInt(strings[0]);
                int cur2 = Integer.parseInt(strings[1]);
                list.add(new int[] { cur1 * flag, cur2});
                if (m == -1) {
                    m = cur2;
                } else {
                    m = (int) (m / gcd(m, cur2) * cur2);
                }
                index = next;
            }
        }
        int mm = 0;
        for (int[] cur : list) {
            mm += (m / cur[1]) * cur[0];
        }
        StringBuilder ans = new StringBuilder();
        if (mm < 0) {
            ans.append("-");
        }
        int vv = (int) gcd(Math.abs(mm), m);
        ans.append(mm / vv).append("/").append(m / vv);
        return sb.toString();
    }

    public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
        Set<Integer> set = new HashSet<>();
        set.add(getLen(p1, p2));
        set.add(getLen(p1, p3));
        set.add(getLen(p1, p4));
        set.add(getLen(p2, p3));
        set.add(getLen(p2, p4));
        set.add(getLen(p3, p4));
        return set.size() == 2;
    }

    public int getLen(int[] p1, int[] p2) {
        return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]);
    }

    public int largestComponentSize(int[] nums) {
        UnionFind unionFind = new UnionFind(100010);
        for (int num : nums) {
            for (int i = 2;i * i <= num; i++) {
                if (num % i == 0) {
                    unionFind.union(num, i);
                    if (i != num / i) {
                        unionFind.union(num, num / i);
                    }
                }
            }
        }
        int ans = 0;
        int[] count = new int[101010];
        for (int num : nums) {
            int root = unionFind.find(num);
            count[root]++;
            ans = Math.max(ans, count[root]);
        }
        return ans;
    }

    public int minimumOperations(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int x : nums) {
            if (x > 0)
                set.add(x);
        }
        return set.size();
    }

    public int maximumGroups(int[] grades) {
        Arrays.sort(grades);
        int ans = 0;
        int index = 1;
        for (int i = 1;i < grades.length;i+=index) {
            index++;
            ans++;
            if (i + index >= grades.length) {
                break;
            }
        }
        return ans;
    }

    public int closestMeetingNode(int[] edges, int node1, int node2) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0;i < edges.length; i++) {
            if (edges[i] != -1) {
                map.putIfAbsent(i, new ArrayList<>());
                map.get(i).add(edges[i]);
            }
        }
        Map<Integer, Tuple> map1 = new HashMap<>();
        map1.put(node1, new Tuple());
        map1.get(node1).setX(0);
        map1.put(node2, new Tuple());
        map1.get(node2).setY(0);
        dfs(map, map1, 1, node1, new int[edges.length], 0);
        dfs(map, map1, 2, node2, new int[edges.length], 0);
        List<Tuple> list = new ArrayList<>();
        for (int k : map1.keySet()) {
            Tuple tuple = map1.get(k);
            if (tuple.x != 0 || tuple.y != 0) {
                list.add(new Tuple(k, Math.max(tuple.x, tuple.y)));
            }
        }
        if (list.size() == 0) {
            return -1;
        }
        list.sort((o1, o2) -> {
            if (o1.y != o2.y) {
                return o1.y - o2.y;
            }
            return o1.x - o2.x;
        });
        return list.get(0).y;
    }

    public void dfs(Map<Integer, List<Integer>> map, Map<Integer, Tuple> map1, int status, int start, int[] vis, int index) {
        if (vis[start] == 1 || map.get(start) == null) {
            return;
        }
        vis[start] = 1;
        for (int x : map.get(start)) {
            if (vis[x] == 0) {
                map1.putIfAbsent(x, new Tuple());
                if (status == 1) {
                    map1.get(x).setX(index + 1);
                } else {
                    map1.get(x).setY(index + 1);
                }
                dfs(map, map1, status, x, vis, index + 1);
            }
        }
    }

    public int maxLevelSum(TreeNode root) {
        int ans = 0;
        int index = 0;
        int v = 0;
        Queue<TreeNode> que = new ArrayDeque<>();
        que.add(root);
        while (!que.isEmpty()) {
            int vv = 0;
            for (int i = que.size() - 1; i >= 0; i--) {
                TreeNode cur = que.poll();
                if (cur.left != null) {
                    que.add(cur.left);
                }
                if (cur.right != null) {
                    que.add(cur.right);
                }
                vv += root.val;
            }
            if (vv > ans) {
                ans = vv;
                v = vv;
            }
            ++index;
        }
        return v;
    }

    public String orderlyQueue(String s, int k) {
        if (k > 1) {
            char[] arr = s.toCharArray();
            Arrays.sort(arr);
            return new String(arr);
        } else {
            String ans = s;
            for (int i = 1;i < s.length(); i++) {
                String cur = s.substring(i) + s.substring(0, i);
                if (cur.compareTo(ans) < 0) {
                    ans = cur;
                }
            }
            return ans;
        }
    }

    public TreeNode addOneRow(TreeNode root, int val, int depth) {
        if (root == null) {
            return null;
        }
        if (depth == 1) {
            return new TreeNode(val, root, null);
        }
        if (depth == 2) {
            root.left = new TreeNode(val, root.left, null);
            root.right = new TreeNode(val, null, root.right);
        } else {
            root.left = addOneRow(root.left, val, depth - 1);
            root.right = addOneRow(root.right, val, depth - 1);
        }
        return root;
    }

    public List<String> stringMatching(String[] words) {
        List<String> ans = new ArrayList<>();
        for (int i = 0;i < words.length; i++) {
            for (int j = 0;j < words.length; j++) {
                if (i != j) {
                    if (words[j].contains(words[i])) {
                        ans.add(words[i]);
                        break;
                    }
                }
            }
        }
        return ans;
    }

    public List<List<Integer>> mergeSimilarItems(int[][] items1, int[][] items2) {
        Map<Integer, Integer> map = new TreeMap<>();
        for (int i = 0;i < items1.length; i++) {
            map.put(items1[i][0], map.getOrDefault(items1[i][0], 0) + items1[i][1]);
        }
        for (int i = 0;i < items2.length; i++) {
            map.put(items2[i][0], map.getOrDefault(items2[i][0], 0) + items2[i][1]);
        }
        List<List<Integer>> list = new ArrayList<>();
        for (int key : map.keySet()) {
            list.add(new ArrayList<>(Arrays.asList(key, map.get(key))));
        }
        return list;
    }

    //j - i == nums[j] - nums[i];
    //j - nums[j] == i - nums[i];
    public long countBadPairs(int[] nums) {
        long n = nums.length;
        Map<Long, Long> map = new HashMap<>();
        for (int i = 0;i < n; i++) {
            map.put((long) (i - nums[i]), map.getOrDefault((long) (i - nums[i]), 0L) + 1);
        }
        long vv = 0;
        for (long key : map.keySet()) {
            if (map.get(key) > 1) {
                long value = map.get(key);
                vv += (value * (value - 1)) / 2;
            }
        }
        long total = (n * (n - 1)) / 2;
        return total - vv;
    }

    public long taskSchedulerII(int[] tasks, int space) {
        long index = 1;
        Map<Integer, Long> map = new HashMap<>();
        for (int i = 0;i < tasks.length; i++) {
            if (map.get(tasks[i]) != null && (index - map.get(tasks[i]) - 1 < space)) {
                index += space - (index - map.get(tasks[i]) - 1);
            }
            map.put(tasks[i], index++);
        }
        return index - 1;
    }

    /**[12,9,7,6,17,19,21]
     * 12 9 7 6 17 19 21
     * 12 9 3 4 6 17 19 21
     * 12 6 3 3 4 6
     * 12 3 3 3 3 4 6
     * 9 3 3 3 3 3 4 6
     * 6 3 3 3 3 3 3 4 6
     * 3 3 3 3 3 3 3 3 4 6
     */
    public long minimumReplacement(int[] nums) {
        if (nums.length == 1) {
            return 0;
        }
        long ans = 0;
        for (int i = nums.length - 2;i >= 0; i--) {
            if (nums[i + 1] < nums[i]) {
                long k = nums[i] / nums[i + 1];
                if (nums[i] % nums[i + 1] != 0) {
                    k++;
                }
                ans += k - 1;
                nums[i] /= k;
            }
        }
        return ans;
    }

    public int arithmeticTriplets(int[] nums, int diff) {
        int ans = 0;
        int n = nums.length;
        for (int i = 0;i < n; i++) {
            for (int j =i + 1;j < n; j++) {
                for (int k = j + 1;k < n; k++) {
                    if (nums[j] - nums[i] == diff && nums[k] - nums[j] == diff) {
                        ans++;
                    }
                }
            }
        }
        return ans;
    }

    public int reachableNodes(int n, int[][] edges, int[] restricted) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        Set<Integer> set = new HashSet<>();
        int ans = 0;
        for (int[] edge : edges) {
            map.putIfAbsent(edge[0], new ArrayList<>());
            map.putIfAbsent(edge[1], new ArrayList<>());
            map.get(edge[0]).add(edge[1]);
            map.get(edge[1]).add(edge[0]);
        }
        for (int k : restricted) {
            set.add(k);
        }
        Queue<Integer> que = new ArrayDeque<>();
        int[] vis = new int[n];
        que.add(0);
        vis[0] = 1;
        while (!que.isEmpty()) {
            int cur = que.poll();
            ans++;
            if (map.get(cur) == null) {
                continue;
            }
            for (int k : map.get(cur)) {
                if (vis[k] == 1 || set.contains(k)) {
                    continue;
                }
                que.add(k);
                vis[k] = 1;
            }
        }
        return ans;
    }

    public boolean validPartition(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 0;
        for (int i = 2;i <= n; i++) {
            if (dp[i - 2] == 1 && nums[i - 2] == nums[i - 1]) {
                dp[i] = 1;
            }
            if (i >= 3 && dp[i - 3] == 1 && nums[i - 2] == nums[i - 3] && nums[i - 1] == nums[i - 2]) {
                dp[i] = 1;
            }
            if (i >= 3 && dp[i - 3] == 1 && nums[i - 1] - 1 == nums[i - 2] && nums[i - 2] - 1 == nums[i - 3]) {
                dp[i] = 1;
            }
        }
        return dp[n] == 1;
    }

    public int longestIdealString(String s, int k) {
        int[] f = new int[26];
        for (int i = 0;i < s.length(); i++) {
            char c = s.charAt(i);
            for (int j = Math.max(c - k, 0);j <= Math.min(25, c + k); j++) {
                f[c] = Math.max(f[c], f[j]);
            }
            f[c]++;
        }
        return Arrays.stream(f).max().getAsInt();
    }

    public String makeLargestSpecial(String s) {
        if (s.length() < 2) {
            return s;
        }
        // 110010
        int cnt = 0;
        List<String> subs = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0, j = 0;i < s.length(); i++) {
            cnt += s.charAt(i) == '1' ? 1 : -1;
            if (cnt == 0) {
                subs.add("1" + makeLargestSpecial(s.substring(j + 1, i)) + "0");
                j = i + 1;
            }
        }
        subs.sort(Comparator.reverseOrder());
        for (String sub : subs) {
            sb.append(sub);
        }
        return sb.toString();
    }

    public String solveEquation(String equation) {
        String s = equation.replaceAll("-", "+-");
        String[] curs = s.split("=");
        int xNum = 0;
        int num = 0;
        int vv = 1;
        for (int i = 0;i < curs.length; i++) {
            String cur = curs[i];
            String[] strings = cur.split("\\+");
            if (i == 1) {
                vv = -1;
            }
            for (String str : strings) {
                if (str.length() == 0) {
                    continue;
                }
                if (str.charAt(str.length() - 1) == 'x') {
                    if (str.length() > 1) {
                        if (str.charAt(0) == '-' && str.length() == 2) {
                            xNum = xNum + vv;
                        } else {
                            xNum += Integer.parseInt(str.substring(0, str.length() - 1)) * vv;
                        }
                    } else {
                        xNum += vv;
                    }
                } else {
                    num += Integer.parseInt(str) * vv;
                }
            }
        }
        if (xNum == 0) {
            return num == 0 ? "Infinite solutions" : "No solution";
        }
        return "x=" + num / xNum * -1;
    }

    public int[][] largestLocal(int[][] grid) {
        int n = grid.length;
        int[][] ans = new int[n - 2][n - 2];
        for (int i = 1;i < n - 1; i++) {
            for (int j = 1;j < n - 1; j++) {
                int cur = grid[i][j];
                cur = Math.max(grid[i - 1][j], cur);
                cur = Math.max(grid[i - 1][j - 1], cur);
                cur = Math.max(grid[i][j - 1], cur);
                cur = Math.max(grid[i + 1][j - 1], cur);
                cur = Math.max(grid[i + 1][j], cur);
                cur = Math.max(grid[i + 1][j + 1], cur);
                cur = Math.max(grid[i][j + 1], cur);
                cur = Math.max(grid[i - 1][j + 1], cur);
                ans[i - 1][j - 1] = cur;
            }
        }
        return ans;
    }

    public int edgeScore(int[] edges) {
        int n = edges.length;
        long[] in = new long[n];
        long maxV = 0;
        for (int i = 0;i < n; i++) {
            in[edges[i]] += i;
            maxV = Math.max(in[edges[i]], maxV);
        }
        int ans = 0;
        for (int i = 0;i < n; i++) {
            if (in[i] == maxV) {
                ans = i;
                break;
            }
        }
        return ans;
    }

    String ans = "zz";
    int[] A = new int[20];
    int[] book = new int[20];

    public String smallestNumber(String pattern) {
        int n = pattern.length();

        permutation(1, n + 1, pattern);
        return ans;
    }

    public void permutation(int s, int n, String pattern) {
        if (s == n + 1) {
            StringBuilder sb = new StringBuilder();
            for (int i = 1;i <= n; i++) {
                sb.append(A[i]);
            }
            boolean check = true;
            for (int i = 1;i < n; i++) {
                if (pattern.charAt(i - 1) == 'I') {
                    if (A[i + 1] < A[i]) {
                        check = false;
                        break;
                    }
                } else {
                    if (A[i + 1] > A[i]) {
                        check = false;
                        break;
                    }
                }
            }
            if (check && sb.toString().compareTo(ans) < 0) {
                ans = sb.toString();
            }
        } else {
            for (int j = 1;j <= n; j++) {
                if (book[j] == 0) {
                    book[j]++;
                    A[s] = j;
                    permutation(s + 1, n, pattern);
                    book[j] = 0;
                }
            }
        }
    }

    static int[][] sum = new int[10][10];

    public int countSpecialNumbers(int n) {
        for (int i = 1;i < 10; i++) {
            for (int j = i;j < 10; j++) {
                int cur = 1;
                for (int k = i;k <= j; k++) {
                    cur *= k;
                }
                sum[i][j] = cur;
            }
        }
        return dp(n) - 1;
    }

    public int dp(int x) {
        int cur = x;
        List<Integer> nums = new ArrayList<>();
        while (cur != 0) {
            nums.add(cur % 10);
            cur /= 10;
        }
        int n = nums.size();
        if (n <= 1) {
            return x + 1;
        }
        int ans = 0;
        // 从高位开始枚举
        for (int i = n - 1, p = 1, s = 0;i >= 0; i--, p++) {

        }
        return 1;
    }

    public int[] answerQueries(int[] nums, int[] queries) {
        int n = nums.length;
        int m = queries.length;
        Arrays.sort(nums);
        for (int i = 1;i < n; i++) {
            nums[i] += nums[i - 1];
        }
        int[] ans = new int[m];
        for (int i = 0;i < m; i++) {
            for (int j = 0;j < n; j++) {
                if (nums[j] < queries[i]) {
                    ans[i] = j + 1;
                }
            }
        }
        return ans;
    }

    public String removeStars(String s) {
        StringBuilder sb = new StringBuilder();
        Deque<Character> stack = new ArrayDeque<>();
        stack.push(s.charAt(0));
        for (int i = 1;i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '*') {
                stack.pop();
            } else {
                stack.push(ch);
            }
        }
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        return sb.reverse().toString();
    }

    public int garbageCollection(String[] garbage, int[] travel) {
        int M = 0, P = 0, G = 0;
        int time = 0;
        for (int i = 0;i < garbage.length; i++) {
            if (garbage[i].contains("M")) {
                M = i;
            }
            if (garbage[i].contains("P")) {
                P = i;
            }
            if (garbage[i].contains("G")) {
                G = i;
            }
            time += garbage[i].length();
        }
        for (int i = 0;i < M; i++) {
            time += travel[i];
        }
        for (int i = 0;i < P; i++) {
            time += travel[i];
        }
        for (int i = 0;i < G; i++) {
            time += travel[i];
        }
        return time;
    }

    int ans0903 = 0;

    public int longestUnivaluePath(TreeNode root) {
        dfs(root);
        return ans0903;
    }

    public int dfs(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = dfs(root.left), right = dfs(root.right);
        int l = 0, r = 0;
        if (root.left != null && root.left.val == root.val) {
            l++;
        }
        if (root.right != null && root.right.val == root.val) {
            r++;
        }
        ans0903 = Math.max(ans0903, l + r);
        return Math.max(l, r);
    }

    public boolean findSubarrays(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 1;i < nums.length; i++) {
            int cur = nums[i] + nums[i - 1];
            map.put(cur, map.getOrDefault(cur, 0) + 1);
        }
        for (int x : map.keySet()) {
            if (map.get(x) >= 2) {
                return true;
            }
        }
        return false;
    }

    public boolean isStrictlyPalindromic(int n) {
        boolean ans = true;
        for (int i = 2;i <= n - 2; i++) {
            String s = conver10ToN(n, i);
            for (int j = 0;j < s.length() / 2; j++) {
                if (s.charAt(j) != s.charAt(s.length() - 1 - j)) {
                    ans = false;
                    break;
                }
            }
        }
        return ans;
    }

    private String conver10ToN(int number, int n) {
        int result = 0;
        StringBuilder sb = new StringBuilder();
        while(number > 0){
            sb.append(number % n);
            number = number / n;
        }
        return sb.toString();
    }

    /**
     * [[1,0,1,0],
     *  [1,0,1,0],
     *  [0,1,1,1],
     *  [0,1,0,0]]
     * 3
     */

    public int maximumRows(int[][] mat, int cols) {
        int m = mat.length;
        int n = mat[0].length;
        int ans = 0;
        for (int i = 0;i < (1 << n); i++) {
            int cnt = 0;
            for (int j = 0;j < n; j++) {
                cnt += ((i >> j) & 1);
            }
            if (cnt != cols) {
                continue;
            }
            cnt = 0;
            for (int j = 0;j < m; j++) {
                boolean flag = true;
                for (int k = 0;k < n; k++) {
                    if (mat[j][k] == 1 && ((i >> k) & 1) == 0) {
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    cnt++;
                }
            }
            ans = Math.max(ans, cnt);
        }
        return ans;
    }

    int[][] fmax = new int[50200][20];

    public void init(int n, int[] nums) {
        for (int i = 1;i <= n; i++) {
            fmax[i][0] = nums[i - 1];
        }
        for (int i = 1;(1 << i) <= n; i++) {
            for (int j = 1;j + (1 << i) - 1 <= n; j++) {
                fmax[j][i] = Math.max(fmax[j][i - 1], fmax[j + (1<<(i - 1))][i - 1]);
            }
        }
    }

    public int max(int l, int r) {
        int k = (int)(Math.log(r-l+1)/Math.log(2));
        return Math.max(fmax[l][k], fmax[r-(1<<k)+1][k]);
    }

    public int maximumRobots(int[] chargeTimes, int[] runningCosts, long budget) {
        int n = chargeTimes.length;
        init(n, chargeTimes);
        int l = 0, r = (int) 10.0;
        long[] sum = new long[n + 1];
        for (int i = 0;i < n; i++) {
            sum[i + 1] = sum[i] + runningCosts[i];
        }
        while (l < r) {
            int mid = (l + r + 1) >> 1;
            if (check(mid, sum, budget)) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        return l;
    }

    public boolean check(int k, long[] sum, long budget) {
        long res = Long.MAX_VALUE;
        for (int i = k;i < sum.length; i++) {
            long max = max(i - k + 1, i);
            long max1 = sum[k] - sum[i - k];
            res = Math.min(res, max + k * max1);
        }
        return res <= budget;
    }

    public boolean checkDistances(String s, int[] distance) {
        Map<Character, List<Integer>> map = new HashMap<>();
        for (int i = 0;i < s.length(); i++) {
            char cur = s.charAt(i);
            map.putIfAbsent(cur, new ArrayList<>());
            map.get(cur).add(i);
        }
        boolean ans = true;
        for (char K : map.keySet()) {
            List<Integer> integers = map.get(K);
            int cur = distance[K - 'a'];
            int i = integers.get(1) - integers.get(0);
            if (cur != i - 1) {
                ans = false;
                break;
            }
        }
        return ans;
    }

    Map<String, Integer> map = new HashMap<>();

    public int numberOfWays(int startPos, int endPos, int k) {
        int d = Math.abs(startPos - endPos);
        if ((d + k) % 2 == 1 || d > k) {
            return 0;
        }

        return dfs1(startPos, k, endPos);
    }

    public int dfs1(int cur, int left, int end) {
        String key = "" + cur + "," + left;
        if (map.get(key) != null) {
            return map.get(key);
        }
        if (Math.abs(cur - end) > left) {
            map.put(key, 0);
            return 0;
        }
        int res = (dfs1(cur - 1, left - 1, end) + dfs1(cur + 1, left + 1, end)) % mod;
        map.put(key, res);
        return res;
    }

    public int longestNiceSubarray(int[] nums) {
        int ans = 0;
        int[] bits = new int[32];
        int l = 0, r = 0;
        while (r < nums.length) {
            int cur = nums[r];
            int index = 0;
            while (cur > 0) {
                bits[index++] += cur & 1;
                cur >>= 1;
            }
            while (!checkBits(bits)) {
                cur = nums[l];
                index = 0;
                while (cur > 0) {
                    bits[index++] -= cur & 1;
                    cur >>= 1;
                }
                l++;
            }
            ans = Math.max(ans, (r - l + 1));
            r++;
        }
        return ans;
    }

    public boolean checkBits(int[] bits) {
        for (int i = 0;i < 32; i++) {
            if (bits[i] > 1) {
                return  false;
            }
        }
        return true;
    }

    public int mostBooked(int n, int[][] meetings) {
        int[] cnt = new int[n];
        Arrays.sort(meetings, Comparator.comparingInt(o -> o[0]));
        PriorityQueue<Integer> idle = new PriorityQueue<>(Comparator.comparingInt(o -> o));
        PriorityQueue<Pair<Long, Integer>> using = new PriorityQueue<>((o1, o2) -> {
            if (!Objects.equals(o1.getKey(), o2.getKey())) {
                return (int) (o1.getKey() - o2.getKey());
            }
            return o1.getValue() - o2.getValue();
        });
        for (int i = 0;i < n; i++) {
            idle.add(i);
        }
        for (int[] m : meetings) {
            int start = m[0], end = m[1];
            while (!using.isEmpty() && using.peek().getKey() <= start) {
                idle.add(using.poll().getValue());
            }
            int id = -1;
            if (!idle.isEmpty()) {
                id = idle.poll();
            } else if (!using.isEmpty()){
                Pair<Long, Integer> peek = using.poll();
                end += peek.getKey() - start;
                id = peek.getValue();
            }
            cnt[id]++;
            using.add(new Pair<>((long) end, id));
        }
        int ans = 0;
        for (int i = 0;i < n; i++) {
            if (cnt[i] > cnt[ans]) {
                ans = i;
            }
        }
        return ans;
    }

    public int findLongestChain(int[][] pairs) {
        TreeSet<Integer>[] lists = new TreeSet[4000];
        for (int i = 0;i < lists.length; i++) {
            lists[i] = new TreeSet<>();
        }
        for (int[] p : pairs) {
            lists[p[1] + 1000].add(p[0]);
        }
        int[][] pp = new int[pairs.length][2];
        int index = 0;
        for (int i = 0;i < lists.length; i++) {
            while (!lists[i].isEmpty()){
                pp[index][0] = i - 1000;
                pp[index++][1] = lists[i - 1000].pollFirst();
            }
        }
        int cur = Integer.MIN_VALUE;
        int tmp = 0;
        for (int[] P : pp) {
            if (cur < P[0]) {
                cur = P[1];
                tmp++;
            }
        }
        return tmp;
    }

    public int mostFrequentEven(int[] nums) {
        Map<Integer, Integer> map = new TreeMap<>();
        int max = 0;
        for (int num : nums) {
            if (num % 2 == 0) {
                map.put(num, map.getOrDefault(num, 0) + 1);
                max = Math.max(max, map.get(num));
            }
        }
        int ans = -1;
        for (int k : map.keySet()) {
            if (map.get(k) == max) {
                ans = k;
                break;
            }
        }
        return ans;
    }

    public int partitionString(String s) {
        int index = 0;
        Set<Character> set = new HashSet<>();
        for (int i = 0;i < s.length(); i++) {
            char cur = s.charAt(i);
            if (set.contains(cur)) {
                index++;
                set.clear();
            }
            set.add(cur);
        }
        return index + 1;
    }

    public int minGroups(int[][] intervals) {
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        PriorityQueue<Integer> que = new PriorityQueue<>();
        que.add(intervals[0][1]);
        for (int i = 1;i < intervals.length; i++) {
            int t = que.peek();
            if (intervals[i][0] <= t) {
                que.add(intervals[i][1]);
            } else {
                que.poll();
                que.add(intervals[i][1]);
            }
        }
        return que.size();
    }

    public ListNode reContruct(ListNode head) {
        ListNode hair = new ListNode(0);
        ListNode p = hair;
        hair.next = head;
        while (head != null) {
            if (head.val % 2 == 0) {
                p.next = head.next;
                head = p.next;
            } else {
                p = p.next;
                head = head.next;
            }
        }
        return hair.next;
    }

    public int[] explorationSupply(int[] station, int[] pos) {
        int[] ans = new int[pos.length];
        for (int i = 0;i < pos.length; i++) {
            int l = 0, r = station.length;
            while (l < r) {
                int mid = (l + r) >> 1;
                if (station[mid] >= pos[i]) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
                if (l == 0) {
                    ans[i] = l;
                } else if(l == station.length) {
                    ans[i] = l - 1;
                } else {
                    if (station[l] - pos[i] >= pos[i] - station[l - 1]) {
                        ans[i] = l - 1;
                    } else {
                        ans[i] = l;
                    }
                }
            }
        }
        return ans;
    }

    public int countDaysTogether(String arriveAlice, String leaveAlice, String arriveBob, String leaveBob) {
        String[] strings1 = arriveAlice.split("-");
        String[] strings2 = leaveAlice.split("-");
        String[] strings3 = arriveBob.split("-");
        String[] strings4 = leaveBob.split("-");
        Set<String> set = new HashSet<>();
        List<Integer> list = Arrays.asList(1,3,5,7,8,10,12);
        int cur = 30;
        if (!strings1[0].equals(strings2[0])) {
            if (Integer.parseInt(strings1[0]) == 2) {
                cur = 28;
            } else if (list.contains(Integer.parseInt(strings1[0]))) {
                cur = 31;
            }
            for (int i = Integer.parseInt(strings1[1]); i <= cur; i++) {
                set.add(Integer.parseInt(strings1[0]) + "-" + i);
            }
            for (int i = Integer.parseInt(strings1[0]) + 1; i <= Integer.parseInt(strings2[0]) - 1; i++) {
                cur = 30;
                if (i == 2) {
                    cur = 28;
                } else if (list.contains(i)) {
                    cur = 31;
                }
                for (int j = 1; j <= cur; j++) {
                    set.add(i + "-" + j);
                }
            }
            for (int i = 1; i <= Integer.parseInt(strings2[1]); i++) {
                set.add(Integer.parseInt(strings2[0]) + "-" + i);
            }
        } else {
            for (int i = Integer.parseInt(strings1[1]);i <= Integer.parseInt(strings2[1]); i++) {
                set.add(Integer.parseInt(strings1[0]) + "-" + i);
            }
        }
        int ans = 0;
        if (!strings3[0].equals(strings4[0])) {
            if (Integer.parseInt(strings3[0]) == 2) {
                cur = 28;
            } else if (list.contains(Integer.parseInt(strings3[0]))) {
                cur = 31;
            }
            for (int i = Integer.parseInt(strings3[1]); i <= cur; i++) {
                String vv = Integer.parseInt(strings3[0]) + "-" + i;
                if (set.contains(vv)) {
                    ans++;
                }
            }
            for (int i = Integer.parseInt(strings3[0]) + 1; i <= Integer.parseInt(strings4[0]) - 1; i++) {
                cur = 30;
                if (i == 2) {
                    cur = 28;
                } else if (list.contains(i)) {
                    cur = 31;
                }
                for (int j = 1; j <= cur; j++) {
                    if (set.contains(i + "-" + j)) {
                        ans++;
                    }
                }
            }
            for (int i = 1; i <= Integer.parseInt(strings4[1]); i++) {
                if (set.contains(Integer.parseInt(strings4[0]) + "-" + i)) {
                    ans++;
                }
            }
        } else {
            for (int i = Integer.parseInt(strings3[1]);i <= Integer.parseInt(strings4[1]); i++) {
                if (set.contains(Integer.parseInt(strings3[0]) + "-" + i)) {
                    ans++;
                }
            }
        }
        return ans;
    }

    public int matchPlayersAndTrainers(int[] players, int[] trainers) {
        List<Integer> list = new ArrayList<>();
        for (int x : trainers) {
            list.add(x);
        }
        Collections.sort(list);
        int ans = 0;
        for (int i = 0;i < players.length; i++) {
            int l = 0, r = list.size();
            if (list.size() == 0) {
                break;
            }
            while (l < r) {
                int mid = (l + r) >> 1;
                if (list.get(mid) >= players[i]) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }
            if (l >= list.size()) {
                continue;
            } else {
                ans++;
                list.remove(l);
            }
        }
        return ans;
    }

    public int[] smallestSubarrays(int[] nums) {
        int[] maxs = new int[nums.length];
        int[] ans = new int[nums.length];
        Map<Integer, List<Integer>> map = new HashMap<>();
        int cur = 0;
        for (int i = nums.length - 1;i >= 0; i--) {
            cur |= nums[i];
            maxs[i] = cur;
        }
        for (int i = 0;i < nums.length; i++) {
            int vv = nums[i];
            int index = 0;
            while (vv > 0) {
                if (vv % 2 == 1) {
                    map.putIfAbsent(index, new ArrayList<>());
                    map.get(index).add(i);
                }
                index++;
                vv >>= 1;
            }
        }
        for (int i = 0;i < nums.length; i++) {
            int max = maxs[i];
            int index = 0;
            int right = i;
            while (max > 0) {
                if (max % 2 == 1) {
                    List<Integer> list = map.get(index);
                    int l = 0, r = list.size();
                    while (l < r) {
                        int mid = (l + r) >> 1;
                        if (list.get(mid) >= i) {
                            r = mid;
                        } else {
                            l = mid + 1;
                        }
                    }
                    right = Math.max(right, list.get(l));
                }
                index++;
                max >>= 1;
            }
            ans[i] = right - i + 1;
        }
        return ans;
    }

    private boolean check(Double[][] cur, int[][] transactions, long k) {
        for (Double[] vv : cur) {
            int index = vv[0].intValue();
            if (k < transactions[index][0]) {
                return false;
            }
            k = k - transactions[index][0] + transactions[index][1];
        }
        return true;
    }

    public int smallestEvenMultiple(int n) {
        if (n == 1) {
            return 2;
        }
        if (n % 2 == 0) {
            return n;
        }
        return n * 2;
    }

    public int longestContinuousSubstring(String s) {
        int n = s.length();
        int ans = 1;
        int cur = 1;
        for (int i = 1;i < n; i++) {
            char vv1 = s.charAt(i);
            char vv2 = s.charAt(i - 1);
            if ((vv1 - 'a') == (vv2 - 'a') + 1) {
                cur++;
            } else {
                ans = Math.max(ans, cur);
                cur = 1;
            }
        }
        ans = Math.max(ans, cur);
        return ans;
    }

    public TreeNode reverseOddLevels(TreeNode root) {
        Queue<TreeNode> que = new ArrayDeque<>();
        List<Integer> list = new ArrayList<>();
        que.add(root);
        int vv = 0;
        while (!que.isEmpty()) {
            for (int i = que.size() - 1; i >= 0; i--) {
                TreeNode cur = que.poll();
                list.add(cur.val);
                if (cur.left != null) {
                    que.add(cur.left);
                }
                if (cur.right != null) {
                    que.add(cur.right);
                }
            }
            vv++;
        }
        int vs = 2;
        for (int i = 1;i < vv; i += 2) {
            int start = vs;
            int end = (int) Math.pow(2, i + 1) - 1;
            while (start < end) {
                int t = list.get(start - 1);
                list.set(start - 1, list.get(end - 1));
                list.set(end - 1, t);
                start++;
                end--;
            }
            vs *= 4;
        }
        que.clear();
        que.add(root);
        int index = 0;
        while (!que.isEmpty()) {
            for (int i = que.size() - 1; i >= 0; i--) {
                TreeNode cur = que.poll();
                cur.val = list.get(index++);
                if (cur.left != null) {
                    que.add(cur.left);
                }
                if (cur.right != null) {
                    que.add(cur.right);
                }
            }
        }
        return root;
    }

    public int[] sumPrefixScores(String[] words) {
        Map<String, Integer> map = new HashMap<>();
        List<String>[] lists = new List[words.length];
        int[] ans = new int[words.length];
        int index = 0;
        for (String word : words) {
            StringBuffer sb = new StringBuffer();
            lists[index] = new ArrayList<>();
            for (int i = 0;i < word.length(); i++) {
                sb.append(word.charAt(i));
                String cur = sb.toString();
                lists[index].add(cur);
                map.put(cur, map.getOrDefault(cur, 0) + 1);
            }
            index++;
        }
        index = 0;
        for (int j = 0;j < words.length; j++) {
            int cur = 0;
            for (String s : lists[index]) {
                cur += map.get(s);
            }
            ans[j] = cur;
            index++;
        }
        return ans;
    }

    public long minimumMoney(int[][] transactions) {
        long totalLose = 0;
        for (int i = 0;i < transactions.length; i++) {
            int cur1 = transactions[i][0];
            int cur2 = transactions[i][1];
            if (cur1 > cur2) {
                totalLose += cur1 - cur2;
            }
        }
        long ans = 0;
        for (int i = 0;i < transactions.length; i++) {
            int cur1 = transactions[i][0];
            int cur2 = transactions[i][1];
            if (cur1 > cur2) {
                totalLose -= (cur1 - cur2);
                ans = Math.max(totalLose + cur1, ans);
                totalLose += (cur1 - cur2);
            } else {
                ans = Math.max(totalLose + cur1, ans);
            }
        }
        return ans;
    }

    public int temperatureTrend(int[] temperatureA, int[] temperatureB) {
        int len = temperatureA.length;
        int[] cur1 = new int[len];
        int[] cur2 = new int[len];
        for (int i = 1;i < len; i++) {
            if (temperatureA[i] > temperatureA[i - 1]) {
                cur1[i] = 1;
            } else if (temperatureA[i] == temperatureA[i - 1]) {
                cur1[i] = 0;
            } else {
                cur1[i] = -1;
            }
        }
        for (int i = 1;i < len; i++) {
            if (temperatureB[i] > temperatureB[i - 1]) {
                cur2[i] = 1;
            } else if (temperatureB[i] == temperatureB[i - 1]) {
                cur2[i] = 0;
            } else {
                cur2[i] = -1;
            }
        }
        int ans = 0;
        int cur = 0;
        for (int i = 1;i < len; i++) {
            if (cur1[i] == cur2[i]) {
                cur++;
            } else {
                ans = Math.max(ans, cur);
                cur = 0;
            }
        }
        ans = Math.max(ans, cur);
        return ans;
    }

    public int transportationHub(int[][] path) {
        int max = 0;
        for (int[] p : path) {
            max = Math.max(max, Math.max(p[0], p[1]));
        }
        int[] in = new int[max + 1];
        int[] out = new int[max + 1];
        for (int[] p : path) {
            int x = p[0];
            int y = p[1];
            in[y]++;
            out[x]++;
        }
        for (int i = 0;i <= max; i++) {
            if (in[i] == max && out[i] == 0) {
                return i;
            }
        }
        return -1;
    }

    public String[] sortPeople(String[] names, int[] heights) {
        int len  = names.length;
        Vs[] vs = new Vs[len];
        for (int i = 0;i < len; i++) {
            vs[i] = new Vs(names[i], heights[i]);
        }
        Arrays.sort(vs, new Comparator<Vs>() {
            @Override
            public int compare(Vs o1, Vs o2) {
                return o2.y - o1.y;
            }
        });
        String[] ans = new String[len];
        for (int i = 0;i < len; i++) {
            ans[i] = vs[i].x;
        }
        return ans;
    }

    class Vs {
        String x;
        int y;

        public Vs(String x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    public int longestSubarray(int[] nums) {
        int max = Arrays.stream(nums).max().getAsInt();
        int len = nums.length;
        int[] vis = new int[len];
        int ans = 0;
        for (int i = 0;i < len; i++) {
            if (nums[i] == max && vis[i] == 0) {
                int l = i - 1, r = i + 1;
                vis[i] = 1;
                while (l >= 0 && (nums[l] & max) == max) {
                    vis[l] = 1;
                    l--;
                }
                while (r < len && (nums[r] & max) == max) {
                    vis[r] = 1;
                    r++;
                }
                ans = Math.max(ans, (r - l - 1));
            }
        }
        return ans;
    }

    public List<Integer> goodIndices(int[] nums, int k) {
        List<Integer> list = new ArrayList<>();
        int n = nums.length;
        int[] sum1 = new int[n];
        int[] sum2 = new int[n];
        int[] cur1 = new int[n];
        int[] cur2 = new int[n];
        for (int i = 1;i < n; i++) {
            cur1[i] = nums[i] > nums[i - 1] ? -1 : 1;
            cur2[i] = nums[i] >= nums[i - 1] ? 1 : -1;
        }
        for (int i = 1;i < n; i++) {
            sum1[i] = sum1[i - 1] + cur1[i];
            sum2[i] = sum2[i - 1] + cur2[i];
        }
        for (int i = k;i < n - k; i++) {
            int vv1 = sum1[i - 1] - sum1[i - k];
            int vv2 = sum2[i + k] - sum2[i + 1];
            if (vv1 == k - 1 && vv2 == k - 1) {
                list.add(i);
            }
        }
        return list;
    }

    int[] fa;

    public int numberOfGoodPaths(int[] vals, int[][] edges) {
        int n = vals.length;
        List<Integer>[] lists = new ArrayList[n];
        Arrays.setAll(lists, e -> new ArrayList<>());
        for (int[] edge : edges) {
            lists[edge[0]].add(edge[1]);
            lists[edge[1]].add(edge[0]);
        }
        fa = new int[n];
        for (int i = 0;i < n; i++) {
            fa[i] = i;
        }
        int ans = n;
        int[] size = new int[n];
        Arrays.fill(size, 1);
        Integer[] ids = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, Comparator.comparingInt(i -> vals[i]));
        for (int x : ids) {
            int fx = find(x);
            for (int y : lists[x]) {
                // 联通块里的最大值
                int fy = find(y);
                if (fx == fy || vals[y] > vals[x]) {
                    continue;
                }
                if (vals[fy] == vals[x]) {
                    ans += size[fx] * size[fy];
                    size[fx] += size[fy];
                }
                fa[fy] = fx;
            }
        }
        return ans;
    }

    public int find(int x) {
        return fa[x] != x ? fa[x] = find(fa[x]) : fa[x];
    }

    public int[] missingTwo(int[] nums) {
        int n = nums.length;
        int[] arr = new int[n + 2];
        int[] ans = new int[2];
        for (int i = 0;i < n; i++) {
            arr[nums[i]] = 1;
        }
        int index = 0;
        for (int i = 0;i < n; i++) {
            if (arr[i] == 0) {
                ans[index++] = i;
            }
        }
        return ans;
    }

    public boolean equalFrequency(String word) {
        for (int i = 0;i < word.length(); i++) {
            Map<Character, Integer> map = new HashMap<>();
            for (int j = 0;j < word.length(); j++) {
                if (i == j) {
                    continue;
                }
                map.put(word.charAt(j), map.getOrDefault(word.charAt(j), 0) + 1);
            }
            Set<Integer> set = new HashSet<>();
            for (char k : map.keySet()) {
                set.add(map.get(k));
            }
            if (set.size() != 1) {
                return false;
            }
        }
        return true;
    }

    /**
     * a1^b1^a1^b2^a1^b3^a1^b4
     * a2^b1^a2^b2^a2^b3^a2^b4
     * a3^b1^a3^b2^a3^b3^a3^b4
     * @param nums1
     * @param nums2
     * @return
     */
    public int xorAllNums(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        if (n == m) {
            return 0;
        }
        if (n % 2 == 0 && m % 2 == 0) {
            return 0;
        }
        if (n % 2 != 0 && m % 2 != 0) {
            int cur1 = 0;
            int cur2 = 0;
            for (int x : nums1) {
                cur1 ^= x;
            }
            for (int x : nums2) {
                cur2 ^= x;
            }
            return cur1 ^ cur2;
        }
        if (n % 2 != 0) {
            int cur = 0;
            for (int x : nums1) {
                cur ^= x;
            }
            return cur ^ nums2[0];
        }
        int cur = 0;
        for (int x : nums2) {
            cur ^= x;
        }
        return cur ^ nums1[0];
    }

    /**
     *
     * cur1[i] + cur2[j] <= diff
     * @param nums1
     * @param nums2
     * @param diff
     * @return
     */
    public long numberOfPairs(int[] nums1, int[] nums2, int diff) {
        int n = nums1.length;
        int[] cur1 = new int[n];
        int[] cur2 = new int[n];
        List<Integer>[] lists = new ArrayList[n];
        List<Integer> list = new ArrayList<>();
        for (int i = 0;i < n; i++) {
            cur1[i] = nums1[i] - nums2[i];
            cur2[i] = nums2[i] - nums1[i];
        }
        for (int i = n - 1;i >= 0; i--) {
            if (i != n - 1) {
                lists[i] = new ArrayList<>(list);
            }
            list.add(cur2[i]);
        }
        int ans = 0;
        for (int i = 0;i < n - 1; i++) {
            Collections.sort(lists[i]);
            int k = cur1[i] - diff;
            int l = 0, r = lists[i].size();
            while (l < r) {
                int mid = (l + r) >> 1;
                if (lists[i].get(mid) <= k) {
                    l = mid + 1;
                } else {
                    r = mid;
                }
            }
            ans += l;
        }
        return ans;
    }

    public int minimizeXor(int num1, int num2) {
        int cur1 = Integer.bitCount(num1);
        int cur2 = Integer.bitCount(num2);
        if (cur1 == cur2) {
            return num1;
        } else {
            int[] c1 = new int[32];
            int[] c2 = new int[32];
            int index = 0;
            int v1 = num1;
            while (v1 > 0) {
                if ((v1 & 1) == 1) {
                    c1[index] = 1;
                }
                index++;
                v1 >>= 1;
            }
            for (int i = 31;i >= 0; i--) {
                if (cur2 > 0 && c1[i] == 1) {
                    cur2--;
                    c2[i] = 1;
                }
            }
            for (int i = 0;i < 32; i++) {
                if (cur2 > 0 && c1[i] == 0) {
                    cur2--;
                    c2[i] = 1;
                }
            }
            int ans = 0;
            int vv = 1;
            for (int i = 0;i < 32; i++) {
                if (c2[i] != 0) {
                    ans += vv;
                }
                vv *= 2;
            }
            return ans;
        }
    }

    public int countTime(String time) {
        String[] strings = time.split(":");
        int cur1 = 0;
        int cur2 = 0;
        if (strings[0].charAt(0) != '?' && strings[0].charAt(1) != '?') {
            cur1 = 1;
        } else if (strings[0].charAt(0) == '?' && strings[0].charAt(1) == '?') {
            cur1 = 24;
        } else if (strings[0].charAt(0) != '?' && strings[0].charAt(1) == '?') {
            if (strings[0].charAt(0) == '0' || strings[0].charAt(0) == '1') {
                cur1 = 10;
            } else {
                cur1 = 3;
            }
        } else {
            int i = strings[0].charAt(1) - '0';
            if (i >= 4) {
                cur1 = 2;
            } else {
                cur1 = 3;
            }
        }
        if (strings[1].charAt(0) != '?' && strings[1].charAt(1) != '?') {
            cur2 = 1;
        } else if (strings[1].charAt(0) == '?' && strings[1].charAt(1) == '?') {
            cur2 = 60;
        } else if (strings[1].charAt(0) != '?' && strings[1].charAt(1) == '?') {
            cur2 = 10;
        } else {
            cur2 = 6;
        }

        return cur1 * cur2;
    }

    public int[] productQueries(int n, int[][] queries) {
        int mod = (int) (1e9 + 7);
        int[] ans = new int[queries.length];
        List<Integer> list = new ArrayList<>();
        int cur = n;
        int vv = 1;
        while (cur > 0) {
            if ((cur & 1) == 1) {
                list.add(vv);
            }
            vv *= 2;
            cur >>= 1;
        }
        long[] sum = new long[list.size() + 1];
        sum[0] = 1;
        for (int i = 0;i < sum.length - 1; i++) {
            sum[i + 1] = sum[i] * list.get(i);
            sum[i + 1] %= mod;
        }
        for (int i = 0;i < queries.length; i++) {
            int l = queries[i][0];
            int r = queries[i][1];
            long res = (sum[r + 1] * quickMod(sum[l], mod - 2, mod)) % mod;
            ans[i] = (int) res;
        }
        return ans;
    }

    public long quickMod(long a, long b, long mod) {
        long ans = 1;
        a = a % mod; // 首先将a约束到c范围之内
        while(b > 0)
        {
            if((b & 1) == 1) {
                ans = (ans * a) % mod;
            }
            b = b >> 1;
            a = (a * a) % mod;
        }
        return ans;
    }

    public int minimizeArrayValue(int[] nums) {
        int n = nums.length;
        int max = Arrays.stream(nums).max().getAsInt();
        int l = 0, r = max;
        while (l < r) {
            int mid = (l + r) >> 1;
            long[] clone = new long[n];
            for (int i = 0;i < n; i++) {
                clone[i] = nums[i];
            }
            for (int i = n - 1;i > 0; i--) {
                if (clone[i] > mid) {
                    long det = clone[i] - mid;
                    clone[i] = mid;
                    clone[i - 1] += det;
                }
            }
            if (clone[0] <= mid) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    public int findMaxK(int[] nums) {
        int ans = -1;
        Set<Integer> set = new HashSet<>();
        for (int x : nums) {
            set.add(x);
        }
        for (int x : nums) {
            if (x < 0) {
                int cur = -x;
                if (set.contains(cur)) {
                    ans = Math.min(ans, cur);
                }
            }
        }
        return ans;
    }

    public int countDistinctIntegers(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int x : nums) {
            set.add(x);
        }
        for (int x : nums) {
            String cur = String.valueOf(x);
            StringBuilder sb = new StringBuilder(cur);
            sb.reverse();
            set.add(Integer.valueOf(sb.toString()));
        }
        return set.size();
    }

    public boolean sumOfNumberAndReverse(int num) {
        for (int i = 140;i <= 140; i++) {
            int cur = i;
            int vv = 0;
            while (cur > 0) {
                int i1 = cur % 10;
                vv += vv * 10 + i1;
                cur /= 10;
            }
            if (i + vv == num) {
                return true;
            }
        }
        return false;
    }

    public long countSubarrays(int[] nums, int minK, int maxK) {
        long ans = 0;
        int last = -1;
        int mini = -1, maxi = -1;
        for (int i = 0;i < nums.length; i++) {
            if (nums[i] == minK) {
                mini = i;
            }
            if (nums[i] == maxK) {
                maxi = i;
            }
            if (nums[i] > maxK || nums[i] < minK) {
                last = i;
            }
            ans += Math.max(Math.min(mini, maxi) - last, 0);
        }
        return ans;
    }

    public int componentValue(int[] nums, int[][] edges) {
        int sum = Arrays.stream(nums).sum();
        int n = nums.length;
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] e : edges) {
            int x = e[0];
            int y = e[1];
            map.putIfAbsent(x, new ArrayList<>());
            map.putIfAbsent(y, new ArrayList<>());
            map.get(x).add(y);
            map.get(y).add(x);
        }
        for (int i = n;i > 0; i--) {
            if (sum % i == 0) {
                if (dfs(map, nums, 0, -1, sum / i) == 0) {
                    return i - 1;
                }
            }
        }
        return 0;
    }

    public int dfs(Map<Integer, List<Integer>> map, int[] nums, int s, int fa, int target) {
        int cur = nums[s];
        if (map.get(s) != null) {
            for (int x : map.get(s)) {
                if (x != fa) {
                    int res = dfs(map, nums, x, s, target);
                    if (res < 0) {
                        return -1;
                    }
                    cur += res;
                }
            }
        }
        if (cur > target) {
            return -1;
        }
        if (cur == target) {
            return 0;
        }
        return cur;
    }

    public boolean haveConflict(String[] event1, String[] event2) {
        int s1 = ss(event1[0]);
        int e1 = ss(event1[1]);
        int s2 = ss(event2[0]);
        int e2 = ss(event2[0]);
        return Math.max(s1, s2) < Math.min(e1, e2);
    }

    public int ss(String str) {
        String[] split = str.split(":");
        return Integer.parseInt(split[0]) * 60 + Integer.parseInt(split[1]);
    }

    public int subarrayGCD(int[] nums, int k) {
        int ans = 0;
        for (int i = 0;i < nums.length; i++) {
            int gcd = nums[i];
            if (gcd == k) {
                ans++;
            }
             for (int j = i + 1;j < nums.length; j++) {
                 gcd = (int) gcd(gcd, nums[j]);
                 if (gcd == k) {
                     ans++;
                 }
             }
        }
        return ans;
    }

    public int partitionDisjoint(int[] nums) {
        int n = nums.length;
        int leftMax = nums[0], leftPos = 0, curMax = nums[0];
        for (int i = 1;i < n - 1; i++) {
            curMax = Math.max(curMax, nums[i]);
            if (nums[i] < leftMax) {
                leftMax = curMax;
                leftPos = i;
            }
        }
        return leftPos + 1;
    }

    public int sumSubarrayMins(int[] arr) {
        int mod = (int) (1e9+7);
        int n = arr.length;
        int[] left = new int[n];
        int[] right = new int[n];
        long ans = 0;
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0;i < n; i++) {
            while (!stack.isEmpty() && arr[stack.peek()] > arr[i]) {
                stack.pop();
            }
            if (stack.isEmpty()) {
                left[i] = -1;
            } else {
                left[i] = stack.peek();
            }
            stack.push(i);
        }
        stack.clear();
        for (int i = n - 1;i >= 0; i--) {
            while (!stack.isEmpty() && arr[stack.peek()] >= arr[i]) {
                stack.pop();
            }
            if (stack.isEmpty()) {
                right[i] = n;
            } else {
                right[i] = stack.peek();
            }
            stack.push(i);
        }
        for (int i = 0;i < n; i++) {
            ans += (long) (i - left[i]) * (right[i] - i) * arr[i];
            ans %= mod;
        }
        return (int) (ans % mod);
    }

    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        Arrays.fill(ans, -1);
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0;i < 2 * n - 1; i++) {
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i % n]) {
                ans[stack.pop()] = nums[i % n];
            }
            stack.push(i % n);
        }
        return ans;
    }

    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] ans = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0;i < n; i++) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                ans[stack.peek()] = i - stack.peek();
                stack.pop();
            }
            stack.push(i);
        }
        return ans;
    }

    public int[] mostCompetitive(int[] nums, int k) {
        int n = nums.length;
        int[] ans = new int[k];
        Deque<Integer> stack = new ArrayDeque<>();
        int p = n - k;
        for (int i = 0;i < n; i++) {
            while (!stack.isEmpty() && p > 0 && nums[i] < nums[stack.peek()]) {
                stack.pop();
                p--;
            }
            stack.push(i);
        }
        while (p > 0) {
            stack.pop();
            p--;
        }
        while (k > 0) {
            k--;
            ans[k] = nums[stack.pop()];
        }
        return ans;
    }

    public int largestRectangleArea(int[] heights) {
        Deque<Integer> stack = new ArrayDeque<>();
        int n = heights.length;
        int[] right = new int[n];
        int[] left = new int[n];
        Arrays.fill(right, n);
        Arrays.fill(left, -1);
        for (int i = 0;i < heights.length; i++) {
            while (!stack.isEmpty() && heights[stack.peek()] > heights[i]) {
                right[stack.pop()] = i;
            }
            if (!stack.isEmpty()) {
                left[i] = stack.peek();
            }
            stack.push(i);
        }
        int ans = 0;
        for (int i = 0;i < n; i++) {
            ans = Math.max((right[i] - left[i] - 1) * heights[i], ans);
        }
        return ans;
    }

    public int maximalRectangle(char[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;
        int[][] val = new int[n][m];
        int ans = 0;
        for (int i = 0;i < n; i++) {
            for (int j = 0;j < m; j++) {
                int cur = matrix[i][j] - '0';
                if (i == 0) {
                    val[i][j] = cur;
                } else {
                    if (cur != 0) {
                        val[i][j] = val[i - 1][j] + 1;
                    }
                }
            }
        }
        for (int i = 0;i < n; i++) {
            int[] left = new int[m];
            int[] right = new int[m];
            Arrays.fill(right, m);
            Arrays.fill(left, -1);
            Deque<Integer> stack = new ArrayDeque<>();
            for (int j = 0;j < m ; j++) {
                while (!stack.isEmpty() && val[i][stack.peek()] > val[i][j]) {
                    right[stack.pop()] = j;
                }
                if (!stack.isEmpty()) {
                    left[j] = stack.peek();
                }
                stack.push(j);
            }
            for (int j = 0;j < m; j++) {
                int w = right[j] - left[j] - 1;
                int area = w * val[i][j];
                ans = Math.max(area, ans);
            }
        }
        return ans;
    }

    public String oddString(String[] words) {
        Map<String, Integer> map = new HashMap<>();
        Map<String, String> map1 = new HashMap<>();
        for (String s : words) {
            char[] chars = s.toCharArray();
            StringBuilder sb = new StringBuilder();
            for (int i = 1; i < chars.length; i++) {
                sb.append(chars[i] - chars[i - 1]).append(",");
            }
            map.put(sb.toString(), map.getOrDefault(sb.toString(), 0) + 1);
            map1.put(sb.toString(), s);
        }
        for (String s : map.keySet()) {
            if (map.get(s) == 1) {
                return map1.get(s);
            }
        }
        return "";
    }

    public List<String> twoEditWords(String[] queries, String[] dictionary) {
        List<String> list = new ArrayList<>();
        for (String s : queries) {
            for (String d : dictionary) {
                int vv = 0;
                for (int i = 0;i < s.length(); i++) {
                    if (s.charAt(i) != d.charAt(i)) {
                        vv++;
                    }
                }
                if (vv <= 2) {
                    list.add(s);
                    break;
                }
            }
        }
        return list;
    }

    public int destroyTargets(int[] nums, int space) {
        Arrays.sort(nums);
        Map<Integer, List<Integer>> map1 = new HashMap<>();
        int max = 0;
        int ans = Integer.MAX_VALUE;
        int n = nums.length;
        for (int i = 0;i < n; i++) {
            int cur = nums[i] % space;
            map1.putIfAbsent(cur, new ArrayList<>());
            map1.get(cur).add(i);
            max = Math.max(max, map1.get(cur).size());
        }
        for (int k : map1.keySet()) {
            List<Integer> integers = map1.get(k);
            if (integers.size() == max) {
                for (int i = 0;i < integers.size(); i++) {
                    ans = Math.min(nums[integers.get(i)], ans);
                }
            }
        }
        return ans;
    }

    private int[] arr;
    private int[] tree;

    public int[] secondGreaterElement(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        int[] tmp = new int[n];
        Arrays.fill(ans, -1);
        Arrays.fill(tmp, -1);
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0;i < n; i++) {
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i]) {
                tmp[stack.pop()] = i;
            }
            stack.push(i);
        }
        this.tree = new int[n * 4 + 1];
        this.arr = nums;
        buildTree(0, n - 1,1);
        for (int i = 0;i < n; i++) {
            if (tmp[i] != -1) {
                ans[i] = queryRight(tmp[i] + 1, nums[i], 0, n - 1, 1);
            }
        }
        return ans;
    }

    private void buildTree(int l, int r, int idx) {
        if (l == r) {
            tree[idx] = arr[l];
        } else {
            int mid = (l + r) >> 1;
            buildTree(l, mid, idx * 2);
            buildTree(mid + 1, r, idx * 2 + 1);
            tree[idx] = Math.max(tree[idx * 2], tree[idx * 2 + 1]);
        }
    }

    private int queryRight(int start, int val, int l, int r, int idx) {
        if (r < start) {
            return -1;
        }
        if (tree[idx] < val) {
            return -1;
        }
        if (l == r) {
            return arr[l];
        } else {
            int mid = (l + r) >> 1;
            int res = -1;
            if (tree[idx * 2] > val) {
                res = queryRight(start, val, l, mid, idx * 2);
            }
            if (res == -1) {
                res = queryRight(start, val, mid + 1, r, idx * 2 + 1);
            }
            return res;
        }
    }

    public int[] secondGreaterElement1(int[] nums) {
        Deque<Integer> stack = new ArrayDeque<>();
        PriorityQueue<int[]> que = new PriorityQueue<>();
        int n = nums.length;
        int[] ans = new int[n];
        Arrays.fill(ans, -1);
        for (int i = 0;i < n; i++) {
            while (!que.isEmpty() && que.peek()[0] < nums[i]) {
                ans[que.poll()[1]] = nums[i];
            }
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i]) {
                que.add(new int[]{nums[stack.peek()], stack.peek()});
                stack.pop();
            }
            stack.push(i);
        }
        return ans;
    }

    public int averageValue(int[] nums) {
        List<Integer> list = new ArrayList<>();
        for (int x : nums) {
            if (x % 3 == 0 && x % 2 == 0) {
                list.add(x);
            }
        }
        int sum = 0;
        for (int x : list) {
            sum += x;
        }
        return list.size() == 0 ? 0 : sum / list.size();
    }

    public List<List<String>> mostPopularCreator(String[] creators, String[] ids, int[] views) {
        int n = creators.length;
        Map<String, Long> map = new HashMap<>();
        Map<String, TreeSet<VVNode>> map1 = new HashMap<>();
        long max = -1;
        for (int i = 0;i < n; i++) {
            String name = creators[i];
            int view = views[i];
            map.put(name, map.getOrDefault(name, 0L) + view);
            max = Math.max(max, map.get(name));
            map1.putIfAbsent(name, new TreeSet<>((o1, o2) -> {
                if (o1.view != o2.view) {
                    return o2.view - o1.view;
                }
                return o1.id.compareTo(o2.id);
            }));
            map1.get(name).add(new VVNode(ids[i], view));
        }
        List<List<String>> ans = new ArrayList<>();
        for (String k : map.keySet()) {
            if (map.get(k) == max) {
                List<String> list = new ArrayList<>();
                TreeSet<VVNode> vvNodes = map1.get(k);
                VVNode first = vvNodes.first();
                list.add(k);
                list.add(first.id);
                ans.add(list);
            }
        }
        return ans;
    }

    class VVNode{
        String id;
        int view;

        public VVNode(String id, int view) {
            this.id = id;
            this.view = view;
        }
    }

    public long makeIntegerBeautiful(long n, int target) {
        if (sumN(n) <= target) {
            return 0;
        }
        long vv = 10;
        while (true) {
            long cur = n;
            long ans = vv - (cur % vv);
            cur += ans;
            if (sumN(cur) <= target) {
                return ans;
            }
            vv *= 10;
        }
    }

    public int sumN(long n) {
        int res = 0;
        while (n > 0) {
            res += n % 10;
            n /= 10;
        }
        return res;
    }

    Map<TreeNode, Integer> height = new HashMap<>();
    int[] res;

    public int[] treeQueries(TreeNode root, int[] queries) {
        getHeight(root);
        res = new int[height.size()];
        height.put(null, 0);
        dfs(root, -1, 0);
        for (int i = 0;i < queries.length; i++) {
            queries[i] = res[queries[i]];
        }
        return queries;
    }

    public void dfs(TreeNode root, int depth, int cur) {
        if (root == null) {
            return;
        }
        ++depth;
        res[root.val] = cur;
        dfs(root.left, depth, Math.max(cur, depth + height.get(root.right)));
        dfs(root.right, depth, Math.max(cur, depth + height.get(root.left)));
    }

    public int getHeight(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int h = 1 + Math.max(getHeight(root.left), getHeight(root.right));
        height.put(root, h);
        return h;
    }

    public boolean arrayStringsAreEqual(String[] word1, String[] word2) {
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        for (String s : word1) {
            sb1.append(s);
        }
        for (String s : word2) {
            sb2.append(s);
        }
        return sb1.toString().equals(sb2.toString());
    }

    public int maxRepeating(String sequence, String word) {
        int s = sequence.length(), w = word.length();
        int[] f = new int[s];
        int ans = 0;
        for (int i = 0;i < s; i++) {
            if (i - w >= 0) {
                if (sequence.substring(i - w, i + 1).equals(word)) {
                    f[i] = f[i - w] + 1;
                }
            }
            ans = Math.max(ans, f[i]);
        }
        return ans;
    }

    public int reachNumber(int target) {
        int sum = 0;
        int index = 0;
        target = Math.abs(target);
        while (sum < target) {
            index++;
            sum += index;
        }
        if (sum == target) {
            return index;
        }

        while ((sum - target) % 2 != 0) {
            index++;
            sum += index;
        }

        return index;
    }

    public boolean parseBoolExpr(String expression) {
        int len = expression.length();
        Deque<Character> stack = new ArrayDeque<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0;i < len; i++) {
            char cur = expression.charAt(i);
            if (cur == '(' || cur == ',') {
                continue;
            } else if (cur == ')') {
                if (!stack.isEmpty() && sb.length() > 0) {
                    char top = stack.pop();
                    boolean flag = sb.charAt(0) == 't';
                    if (top == '!') {
                        flag = !flag;
                    }
                    for (int j = 1;j < sb.length(); j++) {
                        if (top == '&') {
                            flag &= sb.charAt(j) == 't';
                        } else if (top == '|') {
                            flag |= sb.charAt(j) == 't';
                        }
                    }
                    sb.delete(0, sb.length());
                    sb.append(flag ? 't' : 'f');
                }
                continue;
            } else if (cur == 't' || cur == 'f') {
                sb.append(cur);
                continue;
            }
            stack.push(cur);
        }
        return sb.charAt(0) == 't';
    }

    public String interpret(String command) {
        return command
                .replaceAll("\\(\\)", "o")
                .replaceAll("\\(al\\)", "al");
    }

    public int[] applyOperations(int[] nums) {
        int[] ans = new int[nums.length];
        int index = 0;
        for (int i = 0;i < nums.length; i++) {
            if (i + 1 < nums.length && nums[i] == nums[i + 1]) {
                nums[i] *= 2;
                nums[i + 1] = 0;
            }
            if (nums[i] != 0) {
                ans[index++] = nums[i];
            }
        }
        return ans;
    }

    public long maximumSubarraySum(int[] nums, int k) {
        long ans = 0;
        long sum = 0;
        int last = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0;i < k; i++) {
            sum += nums[i];
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        if (map.size() == k) {
            ans = Math.max(ans, sum);
        }
        for (int i = k;i < nums.length; i++) {
            sum += nums[i];
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
            sum -= nums[last];
            int cur = map.get(nums[last]);
            if (cur == 1) {
                map.remove(nums[last]);
            } else {
                map.put(nums[last], cur - 1);
            }
            last++;
            if (map.size() == k) {
                ans = Math.max(ans, sum);
            }
        }
        return ans;
    }

    public long totalCost(int[] costs, int k, int candidates) {
        long ans = 0;
        if (candidates * 2 >= costs.length) {
            Arrays.sort(costs);
            for (int i = 0;i < k; i++) {
                ans += costs[i];
            }
            return ans;
        }
        PriorityQueue<Integer> upQueue = new PriorityQueue<>();
        PriorityQueue<Integer> downQueue = new PriorityQueue<>();
        for (int i = 0;i < candidates; i++) {
            upQueue.add(costs[i]);
        }
        for (int i = costs.length - candidates;i < costs.length; i++) {
            downQueue.add(costs[i]);
        }
        int upIndex = candidates;
        int downIndex = costs.length - candidates - 1;
        while (k > 0 && upIndex <= downIndex) {
            if (upQueue.peek() <= downQueue.peek()) {
                ans += upQueue.poll();
                upQueue.add(costs[upIndex++]);
            } else {
                ans += downQueue.poll();
                downQueue.add(costs[upIndex--]);
            }
            k--;
        }
        List<Integer> list = new ArrayList<>();
        while (!upQueue.isEmpty()) {
            list.add(upQueue.poll());
        }
        while (!downQueue.isEmpty()) {
            list.add(downQueue.poll());
        }
        Collections.sort(list);
        for (int i = 0;i < k; i++) {
            ans += list.get(i);
        }
        return ans;
    }

    public long minimumTotalDistance(List<Integer> robot, int[][] factory) {
        Collections.sort(robot);
        Arrays.sort(factory, Comparator.comparingInt(o -> o[0]));
        int m = robot.size();
        int n = factory.length;
        long[][] dp = new long[n + 1][m + 1];
        for (int i = 1;i <= n; i++) {
            for (int j = 0;j < m; j++) {
                long sum = 0;
                for (int k = 1;k <= Math.min(j, factory[i - 1][1]); k++) {
                    sum += Math.abs(robot.get(j - k) - factory[i - 1][0]);
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][j - k] + sum);
                }
            }
        }
        return dp[n][m];
    }

    public int countConsistentStrings(String allowed, String[] words) {
        int[] chars = new int[26];
        for (int i = 0;i < allowed.length(); i++) {
            int cur = allowed.charAt(i) - 'a';
            chars[cur] = 1;
        }
        int ans = 0;
        for (String word : words) {
            boolean vv = true;
            for (int i = 0;i < word.length(); i++) {
                char c = word.charAt(i);
                if (chars[c - 'a'] == 0) {
                    vv = false;
                    break;
                }
            }
            if (vv) {
                ans++;
            }
        }
        return ans;
    }

    public int orderOfLargestPlusSign(int n, int[][] mines) {
        int[][] gird = new int[n][n];
        for (int i = 0;i < n; i++) {
            Arrays.fill(gird[i], 1);
        }
        for (int[] mine : mines) {
            gird[mine[0]][mine[1]] = 0;
        }
        int[][] cur1 = new int[n][n];
        int[][] cur2 = new int[n][n];
        int[][] cur3 = new int[n][n];
        int[][] cur4 = new int[n][n];
        boolean flag = false;
        for (int i = 0;i < n; i++) {
            for (int j = 0;j < n; j++) {
                if (gird[i][j] == 1) {
                    flag = true;
                    if (j == 0) {
                        cur1[i][j] = 1;
                    } else {
                        cur1[i][j] = cur1[i][j - 1] + 1;
                    }
                }
                if (gird[j][i] == 1) {
                    if (j == 0) {
                        cur3[j][i] = 1;
                    } else {
                        cur3[j][i] = cur3[j - 1][i] + 1;
                    }
                }
            }
        }
        for (int i = 0;i < n; i++) {
            for (int j = n - 1;j >= 0; j--) {
                if (gird[i][j] == 1) {
                    if (j == n - 1) {
                        cur2[i][j] = 1;
                    } else {
                        cur2[i][j] = cur2[i][j + 1] + 1;
                    }
                }
                if (gird[j][i] == 1) {
                    if (j == n - 1) {
                        cur4[j][i] = 1;
                    } else {
                        cur4[j][i] = cur4[j + 1][i] + 1;
                    }
                }
            }
        }
        int ans = flag ? 1 : 0;
        for (int i = 1;i < n - 1; i++) {
            for (int j = 1;j < n - 1; j++) {
                if (gird[i][j] == 1) {
                    int cur = Integer.MAX_VALUE;
                    cur = Math.min(cur, cur1[i][j - 1]);
                    cur = Math.min(cur, cur2[i][j + 1]);
                    cur = Math.min(cur, cur3[i - 1][j]);
                    cur = Math.min(cur, cur4[i + 1][j]);
                    ans = Math.max(cur + 1, ans);
                }
            }
        }
        return ans;
    }

    public int shortestPathAllKeys(String[] grid) {
        int m = grid.length, n = grid[0].length();
        int sx = 0, sy = 0;
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0;i < m; i++) {
            for (int j = 0;j < n; j++) {
                char cur = grid[i].charAt(j);
                if (cur == '@') {
                    sx = i;
                    sy = j;
                } else if (Character.isLowerCase(cur)) {
                    map.putIfAbsent(cur, map.size());
                }
            }
        }
        int[][][] dis = new int[m][n][1 << map.size()];
        for (int i = 0;i < m; i++) {
            for (int j = 0;j < n; j++) {
                Arrays.fill(dis[i][j], -1);
            }
        }
        Queue<int[]> que = new LinkedList<>();
        que.add(new int[]{sx, sy, 0});
        dis[sx][sy][0] = 0;
        while (!que.isEmpty()) {
            int[] cur = que.poll();
            int x = cur[0], y = cur[1];
            for (int[] dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                int mask = cur[2];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    char ch = grid[nx].charAt(ny);
                    if (ch == '#') {
                        continue;
                    }
                    if (ch == '.' || ch == '@') {
                        if (dis[nx][ny][mask] == -1) {
                            dis[nx][ny][mask] = dis[x][y][mask] + 1;
                            que.add(new int[]{nx, ny, mask});
                        }
                    } else if (Character.isLowerCase(ch)) {
                        int idx = map.get(ch);
                        int vv = dis[x][y][mask];
                        mask = mask | (1 << idx);
                        if (((1 << map.size()) - 1) == mask) {
                            return vv + 1;
                        }
                        if (dis[nx][ny][mask] == -1) {
                            dis[nx][ny][mask] = vv + 1;
                            que.add(new int[]{nx, ny, mask});
                        }
                    } else {
                        int idx = map.get(Character.toLowerCase(ch));
                        if ((mask & (1 << idx)) != 0 && dis[nx][ny][mask] == -1) {
                            dis[nx][ny][mask] = dis[x][y][mask] + 1;
                            que.add(new int[]{nx, ny, mask});
                        }
                    }
                }
            }
        }
        return -1;
    }

    public int distinctAverages(int[] nums) {
        List<Integer> list = new ArrayList<>();
        for (int x : nums) {
            list.add(x);
        }
        Collections.sort(list);
        Set<Double> set = new HashSet<>();
        while (list.size() > 0) {
            int a = list.get(0);
            int b = list.get(list.size() - 1);
            set.add((a * 1.0 + b * 1.0) / 2.0);
            list.remove(0);
            list.remove(list.size() - 1);
        }
        return set.size();
    }

    public int countGoodStrings(int low, int high, int zero, int one) {
        int mod = (int) (1e9 + 7);
        int[][] dp = new int[101010][2];
        dp[zero][0] = 1;
        dp[one][1] = 1;
        for (int i = Math.min(zero, one);i <= high; i++) {
            if (i > zero) {
                dp[i][0] += dp[i - zero][0];
                dp[i][0] += dp[i - zero][1];
            }
            if (i > one) {
                dp[i][1] += dp[i - one][1];
                dp[i][1] += dp[i - one][0];
            }
            dp[i][0] %= mod;
            dp[i][1] %= mod;
        }
        long ans = 0;
        for (int i = low;i <= high; i++) {
            ans += dp[i][0];
            ans += dp[i][1];
            ans %= mod;
        }
        return (int) ans;
    }

    public int mostProfitablePath(int[][] edges, int bob, int[] amount) {
        int n = amount.length;
        int[] in = new int[n];
        int[] out = new int[n];
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] e : edges) {
            int x = e[0];
            int y = e[1];
            map.putIfAbsent(x, new ArrayList<>());
            map.putIfAbsent(y, new ArrayList<>());
            map.get(x).add(y);
            map.get(y).add(x);
            in[x]++;in[y]++;
            out[x]++;out[y]++;
        }
        Queue<Integer> que = new ArrayDeque<>();
        int[] par = new int[n];
        for (int i = 0;i < n; i++) {
            par[i] = i;
        }
        int[] vis = new int[n];
        int ans = Integer.MIN_VALUE, index = 0;
        Map<Integer, Integer> timeMap = new HashMap<>();
        que.add(bob);
        timeMap.put(bob, 0);
        vis[bob] = 1;
        while (!que.isEmpty()) {
            boolean flag = false;
            for (int i = que.size();i > 0; i--) {
                int cur = que.poll();
                if (cur == 0) {
                    flag = true;
                    break;
                }
                if (map.get(cur) != null) {
                    List<Integer> list = map.get(cur);
                    for (int x : list) {
                        if (vis[x] == 0) {
                            vis[x] = 1;
                            que.add(x);
                            par[x] = cur;
                        }
                    }
                }
            }
            if (flag) {
                break;
            }
            index++;
        }
        int v = 0;
        while (v != par[v]) {
            timeMap.put(v, index--);
            v = par[v];
        }
        que.clear();
        Arrays.fill(vis, 0);
        que.add(0);
        index = 0;
        vis[0] = 1;
        for (int i = 0;i < n; i++) {
            par[i] = i;
        }
        Map<Integer, Integer> valueMap = new HashMap<>();
        while (!que.isEmpty()) {
            for (int i = que.size();i > 0; i--) {
                int cur = que.poll();
                int value = valueMap.getOrDefault(par[cur], 0);
                int vv = timeMap.get(cur) == null ? Integer.MAX_VALUE : timeMap.get(cur);
                if (index == vv) {
                    value += amount[cur] / 2;
                } else if (index < vv) {
                    value += amount[cur];
                }
                if (in[cur] == 1 && out[cur] == 1 && cur != 0) {
                    ans = Math.max(ans, value);
                }
                valueMap.put(cur, value);
                if (map.get(cur) != null) {
                    List<Integer> list = map.get(cur);
                    for (int x : list) {
                        if (vis[x] == 0) {
                            vis[x] = 1;
                            que.add(x);
                            par[x] = cur;
                        }
                    }
                }
            }
        }
        return ans;
    }

    public String[] splitMessage(String message, int limit) {
        int n = message.length();
        int[] f = new int[n + 1];
        for (int i = 1;i <= n; i++) {
            f[i] = 0;
            for (int x = i;x > 0; x /= 10) {
                f[i]++;
            }
        }

        for (int cur = 1;cur <= n; cur++) {
            int total = 0;
            for (int i = 1, j = 10;i < cur; i *= 10, j *= 10) {
                int each = limit - 3 - f[i] - f[cur];
                total += each * (Math.min(j, cur) - i);
            }

            if (n - total > 0 && n - total <= limit - 3 - 2 * f[cur]) {
                List<String> list = new ArrayList<>();
                total = 0;
                for (int i = 1;i < cur; i++) {
                    int each = limit - 3 - f[i] - f[cur];
                    list.add(message.substring(total, total + each) +
                            "<" + i + "/" + cur + ">");
                    total += each;
                }
                list.add(message.substring(total, n) +
                        "<" + cur + "/" + cur + ">");

                return list.toArray(new String[0]);
            }
        }
        return new String[]{};
    }

    public double[] convertTemperature(double celsius) {
        double[] ans = new double[2];
        ans[0] = celsius + 273.15;
        ans[1] = celsius * 1.8 + 32.00;
        return ans;
    }

    public int subarrayLCM(int[] nums, int k) {
        int n = nums.length;
        int ans = 0;
        for (int i = 0;i < nums.length; i++) {
            int cur = nums[i];
            if (cur == k) {
                ans++;
            }
            for(int j = i + 1;j < nums.length; j++) {
                cur = (int) (cur / gcd(cur, nums[j]) * nums[j]);
                if (cur == k) {
                    ans++;
                } else if (cur > k) {
                    break;
                }
            }
        }
        return ans;
    }

    public int minimumOperations(TreeNode root) {
        Queue<TreeNode> que = new ArrayDeque<>();
        que.add(root);
        int ans = 0;
        while (!que.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            for (int i = que.size();i > 0; i--) {
                TreeNode cur = que.poll();
                list.add(cur.val);
                if (cur.left != null) {
                    que.add(cur.left);
                }
                if (cur.right != null) {
                    que.add(cur.right);
                }
            }
            if (list.size() != 1) {
                ans += getMinSwaps(list);
            }
        }
        return ans;
    }

    public int getMinSwaps(List<Integer> list) {
        List<Integer> list1 = new ArrayList<>(list);
        Collections.sort(list1);
        Map<Integer, Integer> map = new HashMap<>();
        int n = list1.size();
        for (int i = 0;i < n; i++) {
            map.put(list1.get(i), i);
        }
        int loop = 0;
        int[] vis = new int[n];
        for (int i = 0;i < n; i++) {
            if (vis[i] == 0) {
                int j = i;
                while (vis[j] == 0) {
                    vis[j] = 1;
                    j = map.get(list.get(j));
                }
                loop++;
            }
        }
        return n - loop;
    }

    public String customSortString(String order, String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 1;i <= order.length(); i++) {
            map.put(order.charAt(i - 1), i);
        }
        List<Character> list = new ArrayList<>();
        for (int i = 0;i < s.length(); i++) {
            list.add(s.charAt(i));
        }
        list.sort((o1, o2) -> {
            int cur1 = map.getOrDefault(o1, 0);
            int cur2 = map.getOrDefault(o2, 0);
            return cur1 - cur2;
        });
        StringBuilder stringBuilder = new StringBuilder();
        for (char x : list) {
            stringBuilder.append(x);
        }
        return stringBuilder.toString();
    }

    public int unequalTriplets(int[] nums) {
        int ans = 0;
        int n = nums.length;
        for (int i = 0;i < n; i++) {
            for (int j = i + 1;j < n; j++) {
                for (int k = j + 1;k < n; k++) {
                    if (nums[i] == nums[j] && nums[j] == nums[k]) {
                        ans++;
                    }
                }
            }
        }
        return ans;
    }

    List<Integer> list1;

    public List<List<Integer>> closestNodes(TreeNode root, List<Integer> queries) {
        list1 = new ArrayList<>();
        List<List<Integer>> ans = new ArrayList<>();
        preDfs(root);
        for (int x : queries) {
            List<Integer> cur = new ArrayList<>();
            int l = 0, r = list1.size();
            while (l < r) {
                int mid = (l + r) >> 1;
                if (list1.get(mid) > x) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }
            if (l - 1 >= 0 && l - 1 < list1.size()) {
                cur.add(list1.get(l - 1));
            } else {
                cur.add(-1);
            }
            l = 0;r = list1.size();
            while (l < r) {
                int mid = (l + r) >> 1;
                if (list1.get(mid) >= x) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }
            if (l >= 0 && l < list1.size()) {
                cur.add(list1.get(l));
            } else {
                cur.add(-1);
            }

            ans.add(cur);
        }
        return ans;
    }

    public void preDfs(TreeNode root) {
        if (root == null) {
            return ;
        }
        preDfs(root.left);
        list1.add(root.val);
        preDfs(root.right);
    }

    public double champagneTower(int poured, int query_row, int query_glass) {
        double[] row = {poured};
        for (int i = 1;i <= query_row; i++) {
            double[] newRow = new double[i + 1];
            for (int j = 0;j < i; j++) {
                double v = row[j];
                if (v > 1) {
                    newRow[j] = (v - 1) / 2;
                    newRow[j + 1] = (v - 1) / 2;
                }
            }
            row = newRow;
        }
        return Math.min(1, row[query_glass]);
    }

    public int nthMagicalNumber(int n, int a, int b) {
        int mod = (int) (1e9 + 7);
        long lcm = a / gcd(a, b) * b;
        long l = 0, r = (long) Math.min(a, b) * n;
        while (l < r) {
            long mid = (l + r) >> 1;
            if (mid / a + mid / b - mid / lcm >= n) {
                 r = mid;
            } else {
                l = mid + 1;
            }
        }
        return (int) (l % mod);
    }

    public int numberOfCuts(int n) {
        if (n == 1) {
            return 0;
        }
        if (n % 2 == 0) {
            return n / 2;
        }
        return n;
    }

    public int[][] onesMinusZeros(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[] zeroM = new int[m];
        int[] oneM = new int[m];
        int[] zeroN = new int[n];
        int[] oneN = new int[n];
        for (int i = 0;i < m; i++) {
            int a1 = 0;
            int a2 = 0;
            for (int j = 0;j < n; j++) {
                if (grid[i][j] == 1) {
                    a1++;
                } else {
                    a2++;
                }
            }
            zeroM[i] = a2;
            oneM[i] = a1;
        }
        for (int j = 0;j < n; j++) {
            int a1 = 0;
            int a2 = 0;
            for (int i = 0;i < m; i++) {
                if (grid[i][j] == 1) {
                    a1++;
                } else {
                    a2++;
                }
            }
            zeroN[j] = a2;
            oneN[j] = a1;
        }
        int[][] ans = new int[m][n];
        for (int i = 0;i < m; i++) {
            for (int j = 0;j < n; j++) {
                ans[i][j] = oneM[i] + oneN[j] - zeroM[i] - zeroN[j];
            }
        }
        return ans;
    }

    public int bestClosingTime(String customers) {
        int ans = 0;
        int n = customers.length();
        int[] cur1 = new int[n];
        int[] cur2 = new int[n];
        int v = 0;
        for (int i = 1;i <= n; i++) {
            char ch = customers.charAt(i - 1);
            if (ch == 'N') {
                v++;
            }
            cur1[i] = v;
        }
        v = 0;
        for (int i = n - 1;i >= 0; i--) {
            char ch = customers.charAt(i);
            if (ch == 'Y') {
                v++;
            }
            cur2[i] = v;
        }
        int min = Integer.MAX_VALUE;
        for (int i = 0;i <= n; i++) {
            if (min > cur1[i] + cur2[i]) {
                min = cur1[i] + cur2[i];
                ans = i;
            }
        }
        return ans;
    }

    public int pivotInteger(int n) {
        int cur = 0;
        for (int i = 1;i <= n; i++) {
            cur += i;
        }
        int vv = 0;
        for (int i = 1;i <= n; i++) {
            vv += i;
            if (vv == cur) {
                return i;
            }
            cur -= vv;
        }
        return -1;
    }

    public int appendCharacters(String s, String t) {
        int n = t.length();
        int m = s.length();
        int i = 0, j = 0;
        while (i < n && j < m) {
            if (s.charAt(j) == t.charAt(i)) {
                i++;
            }
            j++;
        }
        return n - i;
    }

    public ListNode removeNodes(ListNode head) {
        ListNode hair = new ListNode(0);
        hair.next = head;

        List<Integer> list = new ArrayList<>();
        ListNode p = head;
        while (p != null) {
            list.add(p.val);
            p = p.next;
        }
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0;i < list.size(); i++) {
            while (!stack.isEmpty() && list.get(stack.peek()) < list.get(i)) {
                stack.poll();
            }
            stack.push(i);
        }
        Set<Integer> set = new HashSet<>(stack);
        p = head;
        ListNode q = hair;
        int index = 0;
        while (p != null) {
            if (set.contains(index)) {
                p = p.next;
                q = q.next;
            } else {
                q.next = p.next;
                p = q.next;
            }
            index++;
        }
        return hair.next;
    }

    /**
     * [2,5,1,4,3,6]
     * 1
     * @param nums
     * @param k
     * @return
     */
    public int countSubarrays(int[] nums, int k) {
        int n = nums.length;
        int ans = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0;i < nums.length; i++) {
            if (nums[i] == k) {
                ans++;
                int cur = 0;
                for (int j = i + 1;j < n; j++) {
                    if (nums[j] > k) {
                        cur++;
                    } else {
                        cur--;
                    }
                    if (cur == 1 || cur == 0) {
                        ans++;
                    }
                    map.put(cur, map.getOrDefault(cur, 0) + 1);
                }
                cur = 0;
                for (int j = i - 1;j >= 0; j--) {
                    if (nums[j] > k) {
                        cur++;
                    } else {
                        cur--;
                    }
                    if (cur == 1 || cur == 0) {
                        ans++;
                    }
                        ans += map.getOrDefault(-cur, 0);
                        ans += map.getOrDefault(-cur + 1, 0);

                }
                break;
            }
        }
        return ans;
    }

    public int countPalindromes(String s) {
        int mod = (int) (1e9 + 7);
        long ans = 0;
        char[] chars = s.toCharArray();
        int[] pre = new int[10], suf = new int[10];
        int[][] pre2 = new int[10][10], suf2 = new int[10][10];
        for (int i = chars.length - 1;i >= 0; i--) {
            int d = chars[i] - '0';
            for (int j = 0;j <= 9; j++) {
                suf2[d][j] += suf[j];
            }
            suf[d]++;
        }
        for (char d : chars) {
            int cur = d - '0';
            --suf[cur];
            for (int j = 0;j < 10; j++) {
                suf2[cur][j] -= suf[j];
            }
            for (int j = 0;j < 10; j++) {
                for (int k = 0;k < 10; k++) {
                    ans += (long) pre2[j][k] * suf2[j][k];
                }
            }
            for (int j = 0;j < 10; j++) {
                pre2[cur][j] += pre[j];
            }
            pre[cur]++;
        }
        return (int) (ans % mod);
    }

    public int[] minOperations(String boxes) {
        int n = boxes.length();
        int[] ans = new int[n];
        for (int i = 0;i < n; i++) {
            int cur = 0;
            for (int j = 0;j < n; j++) {
                if (boxes.charAt(j) != 0) {
                    cur += Math.abs(j - i);
                }
            }
            ans[i] = cur;
        }
        return ans;
    }

    public static void main(String[] args) {
        ListNode listNode = new ListNode(5);
        listNode.next = new ListNode(2);
        listNode.next.next = new ListNode(13);
        listNode.next.next.next = new ListNode(3);
        listNode.next.next.next.next = new ListNode(8);
        new NewMain().removeNodes(listNode);
    }

     class Tuple {
        int x, y;

        public Tuple() {
            this.x = -1;
            this.y = -1;
        }

         public Tuple(int x, int y) {
            this.x = x;
            this.y = y;
         }

         public int getX() {
             return x;
         }

         public void setX(int x) {
             this.x = x;
         }

         public int getY() {
             return y;
         }

         public void setY(int y) {
             this.y = y;
         }
     }

    class PP {
        String v1;
        int v2;

        public PP(String v1, int v2) {
            this.v1 = v1;
            this.v2 = v2;
        }
    }

    class UnionFind {
        int[] parent;
        int[] rank;


        public UnionFind(int n) {
            parent = new int[n + 1];
            for (int i = 0;i <= n; i++) {
                parent[i] = i;
            }
            rank = new int[n + 1];
        }

        public void union(int x, int y) {
            int a = find(x);
            int b = find(y);
            if (a == b) {
                return ;
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

        public int find(int x) {
            if (parent[x] == x) {
                return x;
            }
            return parent[x] = find(parent[x]);
        }
    }
}
