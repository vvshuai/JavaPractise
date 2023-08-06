package com.vvs.leetcode;

import org.apache.commons.lang3.ObjectUtils;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class NewNewMain {

    public boolean[] distanceLimitedPathsExist(int n, int[][] edgeList, int[][] queries) {
        int[] par = new int[n];
        for (int i = 0;i <= n; i++) {
            par[i] = i;
        }
        Arrays.sort(edgeList, Comparator.comparingInt(o -> o[2]));
        Integer[] index = IntStream.range(0, queries.length).boxed().toArray(Integer[]::new);
        Arrays.sort(index, Comparator.comparingInt(a -> queries[a][2]));
        boolean[] ans = new boolean[queries.length];
        int k = 0;
        for (int i : index) {
            while (k < edgeList.length && edgeList[k][2] < queries[i][2]) {
                merge(par, edgeList[k][0], edgeList[k][1]);
                k++;
            }
            ans[i] = find(par, queries[i][0]) == find(par, queries[i][1]);
        }
        return ans;
    }

    public void merge(int[] par, int x, int y) {
        x = find(par, x);
        y = find(par, y);
        par[y] = x;
    }

    public int find(int[] par, int x) {
        if (par[x] == x) {
            return x;
        }
        return par[x] = find(par, x);
    }

    public int beautySum(String s) {
        int ans = 0;
        int n = s.length();
        for (int i = 0;i < n; i++) {
            int[] cnt = new int[26];
            int max = Integer.MIN_VALUE;
            for (int j = i;j < n; j++) {
                int v = s.charAt(j) - 'a';
                max = Math.max(++cnt[v], max);
                int min = Integer.MAX_VALUE;
                for (int k = 0;k < 26; k++) {
                    if (cnt[k] > 0) {
                        min = Math.min(cnt[k], min);
                    }
                }
                ans += max - min;
            }
        }
        return ans;
    }

    public int minOperations(int[] nums) {
        int n = nums.length;
        int ans = 0;
        for (int i = 1;i < n; i++) {
            if (nums[i] <= nums[i - 1]) {
                ans += nums[i - 1] - nums[i] + 1;
                nums[i] = nums[i - 1] + 1;
            }
        }
        return ans;
    }

    public char repeatedCharacter(String s) {
        int[] chars = new int[26];
        for(int i = 0;i < s.length(); i++) {
            char cur = s.charAt(i);
            if (chars[cur++] == 1) {
                return cur;
            }
        }
        return '0';
    }

    public int countDigits(int num) {
        int ans = 0, cur = num;
        while (cur > 0) {
            if (num % (cur % 10) == 0) {
                ans++;
            }
            cur /= 10;
        }
        return ans;
    }

    public int distinctPrimeFactors(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int x : nums) {
            for (int i = 2;i <= Math.sqrt(x); i++) {
                if (x % i == 0) {
                    while (x % i == 0) {
                        x /= i;
                        set.add(i);
                    }
                }
            }
            if (x != 1) {
                set.add(x);
            }
        }
        return set.size();
    }

    public int minimumPartition(String s, int k) {
        char[] chars = s.toCharArray();
        int ans = 1;
        long cur = 0;
        for (char c : chars) {
            int num = c - '0';
            if (num > k) {
                return -1;
            }
            if (cur * 10 + num > k) {
                cur = num;
                ans++;
                continue;
            }
            cur = cur * 10 + num;
        }
        return ans;
    }

    private static final int MX = (int) 1e6;
    private static final int[] primes = new int[101010];

    static {
        boolean[] np = new boolean[MX + 1];
        Arrays.fill(primes, MX + 1);
        int cur = 0;
        for (int i = 2;i <= MX; i++) {
            if (!np[i]) {
                primes[cur++] = i;
                for (int j = i;j <= MX / i; j++) {
                    np[i * j] = true;
                }
            }
        }
    }

    public int[] closestPrimes(int left, int right) {
        int p = -1, q = -1;
        for (int i = lower_bound(primes, left);i + 1 < primes.length && primes[i + 1] <= right; i++) {
            if (q < 0 || primes[i + 1] - primes[i] < q - p) {
                p = primes[i];
                q = primes[i + 1];
            }
        }
        return new int[]{p, q};
    }

    private int lower_bound(int[] nums, int target) {
        int l = 0, r = nums.length;
        while (l < r) {
            int mid = (l + r) >> 1;
            if (nums[mid] >= target) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    public int prefixCount(String[] words, String pref) {
        int ans = 0;
        for (String word : words) {
            if (word.startsWith(pref)) {
                ans++;
            }
        }
        return ans;
    }

    public boolean areSentencesSimilar(String sentence1, String sentence2) {
        String[] words1 = sentence1.split(" ");
        String[] words2 = sentence2.split(" ");
        int i = 0, j = 0;
        while (i < words1.length && i < words2.length && words2[i].equals(words1[i])) {
            i++;
        }
        while (j < words1.length - i && j < words2.length - i
                && words2[words2.length - j - 1].equals(words1[words1.length - j - 1])) {
            j++;
        }
        return i + j == Math.min(words2.length, words1.length);
    }

    public boolean evaluateTree(TreeNode root) {
        if (root.left == null) {
            return root.val == 1;
        }
        if (root.val == 2) {
            return evaluateTree(root.left) || evaluateTree(root.right);
        }
        return evaluateTree(root.left) && evaluateTree(root.right);
    }

    public long pickGifts(int[] gifts, int k) {
        long ans = 0;
        PriorityQueue<Integer> queue = new PriorityQueue<>((o1, o2) -> o2 - o1);
        for (int x : gifts) {
            queue.add(x);
        }
        while (k-- > 0) {
            int cur = queue.poll();
            int sqrt = (int) Math.sqrt(cur);
            ans += (cur - sqrt);
            queue.add(sqrt);
        }
        return ans;
    }

    public int balancedString(String s) {
        int[] cnt = new int[26];
        char[] chars = s.toCharArray();
        int n = chars.length, m = n / 4;
        for (char ch : chars) {
            cnt[ch - 'A']++;
        }
        if (cnt['Q' - 'A'] == m
                && cnt['W' - 'A'] == m
                && cnt['E' - 'A'] == m
                && cnt['R' - 'A'] == m) {
            return 0;
        }
        int ans = n;
        for (int l = 0, r = 0;r < n; r++) {
            cnt[chars[r] - 'A']--;
            while (cnt['Q' - 'A'] <= m
                    && cnt['W' - 'A'] <= m
                    && cnt['E' - 'A'] <= m
                    && cnt['R' - 'A'] <= m) {
                ans = Math.min(ans, r - l + 1);
                cnt[chars[l] - 'A']++;
                l++;
            }
        }
        return ans;
    }

    public int longestWPI(int[] hours) {
        int n = hours.length, ans = 0, s = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 1;i <= n; i++) {
            s += (hours[i - 1] > 8 ? 1 : -1);
            if (s > 0) {
                ans = i;
            } else {
                if (map.containsKey(s - 1)) {
                    ans = Math.max(ans, i - map.get(s - 1));
                }
                if (!map.containsKey(s)) {
                    map.put(s, i);
                }
            }
        }
        return ans;
    }

    public int minMaxDifference(int num) {
        String cur = String.valueOf(num);
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        char vv = '#';
        for (int i = 0;i < cur.length(); i++) {
            char c = cur.charAt(i);
            if (c != '9') {
                if (vv == '#') {
                    vv = c;
                }
                if (vv == c) {
                    sb1.append(vv);
                    continue;
                }
            }
            sb1.append(c);
        }
        vv = cur.charAt(0);
        for (int i = 0;i < cur.length(); i++) {
            char c = cur.charAt(i);
            if (vv == c) {
                sb2.append('0');
            } else {
                sb2.append(c);
            }
        }
        return Integer.parseInt(sb1.toString()) - Integer.parseInt(sb2.toString());
    }

    public int minimizeSum(int[] nums) {
        Arrays.sort(nums);
        int a1 = nums[nums.length - 3] - nums[0];
        int a2 = nums[nums.length - 1] - nums[2];
        int a3 = nums[nums.length - 2] - nums[1];
        return Math.min(Math.min(a1, a2), a3);
    }

    public int minOperations(int n) {
        int ans = 0;
        while (n > 0) {
            ans += ((n & 1) == 1) ? 1 : 0;
            n >>= 1;
        }
        return ans;
    }

    public String[] getFolderNames(String[] names) {
        String[] ans = new String[names.length];
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0;i < names.length; i++) {
            String name = names[i];
            if (map.containsKey(name)) {
                int index = map.get(name) + 1;
                while (map.containsKey(name + "(" + index + ")")) {
                    index++;
                }
                map.put(name, index);
                map.put(name + "(" + index + ")", 0);
                ans[i] = name + "(" + index + ")";
            } else {
                map.put(name, 0);
                ans[i] = name;
            }
        }
        return ans;
    }

    public int countTriplets(int[] nums) {
        int[] cnt = new int[1 << 16];
        for (int x : nums) {
            for (int y : nums) {
                cnt[x & y]++;
            }
        }
        int ans = 0;
        for (int x : nums) {
            for (int mask = 0;mask < nums.length; mask++) {
                if ((x & mask) == 0) {
                    ans += cnt[mask];
                }
            }
        }
        return ans;
    }

    public int splitNum(int num) {
        String cur = String.valueOf(num);
        char[] chars = cur.toCharArray();
        Arrays.sort(chars);
        StringBuilder cur1 = new StringBuilder();
        StringBuilder cur2 = new StringBuilder();
        for (int i = 0;i < chars.length; i++) {
            if (i % 2 == 0) {
                cur1.append(chars[i]);
            } else {
                cur2.append(chars[i]);
            }
        }
        int v1 = cur1.length() > 0 ? Integer.parseInt(cur1.toString()) : 0;
        int v2 = cur2.length() > 0 ? Integer.parseInt(cur2.toString()) : 0;
        return v1 + v2;
    }

    public long coloredCells(int n) {
        if (n == 1) {
            return 1;
        }
        long vv = 1;
        long ans = 1;
        for (int i = 2;i <= n; i++) {
            ans += vv;
            vv += 2;
            ans += vv;
        }
        return ans;
    }

    public int countWays(int[][] ranges) {
        int[][] ans = merge(ranges);
        int n = ans.length;
        int mod = (int) (1e9 + 7);
        long vv = 1;
        for (int i = 0;i < n; i++) {
            vv *= 2;
            vv %= mod;
        }
        return (int) vv;
    }

    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][2];
        }
        Arrays.sort(intervals, new Comparator<int[]>() {
            public int compare(int[] interval1, int[] interval2) {
                return interval1[0] - interval2[0];
            }
        });
        List<int[]> merged = new ArrayList<int[]>();
        for (int i = 0; i < intervals.length; ++i) {
            int L = intervals[i][0], R = intervals[i][1];
            if (merged.size() == 0 || merged.get(merged.size() - 1)[1] < L) {
                merged.add(new int[]{L, R});
            } else {
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], R);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }

    public int passThePillow(int n, int time) {
        int ans = 1;
        int dir = 1;
        while (time-- > 0) {
            ans += dir;
            if (ans > n) {
                dir = -1;
                ans--;
                ans--;
            } else if (ans <= 0) {
                dir = 1;
                ans++;
                ans++;
            }
        }
        return ans;
    }

    public long kthLargestLevelSum(TreeNode root, int k) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<Long> list = new ArrayList<>();
        queue.add(root);
        int vv = 0;
        while (!queue.isEmpty()) {
            long cur = 0;
            for (int i = queue.size() - 1; i >= 0; i--) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
                cur += node.val;
            }
            vv++;
            list.add(cur);
        }
        list.sort(Comparator.reverseOrder());
        return vv < k ? -1 : list.get(k - 1);
    }

    public int minimumDeletions(String s) {
        int n = s.length(), b = 0;
        int[] dp = new int[n + 1];
        for (int i = 1;i <= n; i++) {
            if (s.charAt(i - 1) == 'b') {
                dp[i] = dp[i - 1];
                b++;
            } else {
                dp[i] = Math.min(dp[i - 1] + 1, b);
            }
        }
        return dp[n];
    }

    public String[] findLongestSubarray(String[] array) {
        int n = array.length;
        int[] s = new int[n + 1];
        for (int i = 0;i < n; i++) {
            s[i + 1] = s[i] + (Character.isLetter(array[i].charAt(0)) ? 1 : - 1);
        }
        int begin = 0, end = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0;i <= n; i++) {
            int j = map.getOrDefault(s[i], -1);
            if (j < 0) {
                map.put(s[i], j);
            } else if (i - j > end - begin) {
                begin = j;
                end = i;
            }
        }
        return Arrays.copyOfRange(array, begin, end);
    }

    public int findMinimumTime(int[][] tasks) {
        Arrays.sort(tasks, (o1, o2) -> o2[1] - o1[1]);
        int ans = 0;
        boolean[] run = new boolean[2023];
        for (int[] t : tasks) {
            int start = t[0], end = t[1], d = t[2];
            for (int i = start;i <= end; i++) {
                if (run[i]) {
                    d--;
                }
            }
            for (int i = end;d > 0; i--) {
                if (!run[i]) {
                    run[i] = true;
                    d--;
                    ans++;
                }
            }
        }
        return ans;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null) {
            return right;
        }
        if (right == null) {
            return left;
        }
        return root;
    }

    public int minNumberOfHours(int initialEnergy, int initialExperience,
                                int[] energy, int[] experience) {
        int sum = Arrays.stream(energy).sum();
        int ans = initialEnergy > sum ? 0 : sum + 1 - initialEnergy;
        for (int e : experience) {
            if (initialExperience <= e) {
                ans += (e - initialExperience) + 1;
                initialExperience += (e - initialExperience) + 1;
            }
            initialExperience += e;
        }
        return ans;
    }

    public int minimumEffort(int[][] tasks) {
        Arrays.sort(tasks, (o1, o2) -> {
            int v1 = o1[1] - o1[0];
            int v2 = o2[1] - o2[0];
            return v2 - v1;
        });
        int p = 0, sum = 0;
        for (int[] task : tasks) {
            p = Math.max(p, sum + task[1]);
            sum += task[0];
        }
        return p;
    }

    public int brokenCalc(int startValue, int target) {
        int ans = 0;
        while (target > startValue) {
            ans++;
            if (target % 2 == 1) {
                target++;
            } else {
                target >>= 1;
            }
        }
        return ans + startValue - target;
    }

    public int maximizeGreatness(int[] nums) {
        Arrays.sort(nums);
        int i = 0;
        for (int x : nums) {
            if (x > nums[i]) {
                i++;
            }
        }
        return i;
    }

    public long findScore(int[] nums) {
        int n = nums.length;
        Integer[] ids = IntStream.range(0, n)
                        .boxed().toArray(Integer[]::new);
        Arrays.sort(ids, (i, j) -> nums[i] - nums[j]);

        long ans = 0;
        boolean[] vis = new boolean[n + 2]; // 保证下标不越界
        for (int i : ids)
            if (!vis[i + 1]) { // 避免 -1，偏移一位
                vis[i] = vis[i + 2] = true;
                ans += nums[i];
            }
        return ans;
    }

    public int maxWidthOfVerticalArea(int[][] points) {
        Arrays.sort(points, Comparator.comparingInt(o -> o[0]));
        int ans = 0;
        for (int i = 1;i < points.length; i++) {
            ans = Math.max(ans, points[i][0] - points[i - 1][0]);
        }
        return ans;
    }

    boolean flag0331 = false;
    int[][] dirs = {
            {1, 2}, {1, -2},
            {-1, 2}, {-1, -2},
            {2, 1}, {2, -1},
            {-2, 1}, {-2, -1}
    };

    public boolean checkValidGrid(int[][] grid) {
        dfs(0, 0, 0, grid.length * grid.length - 1, grid);
        return flag0331;
    }

    public void dfs(int x, int y, int cur, int end, int[][] grid) {
        if (cur > end) {
            return;
        }
        if (grid[x][y] == end) {
            flag0331 = true;
            return;
        }
        for (int[] dir : dirs) {
            int nx = x + dir[0];
            int ny = y + dir[1];
            if (nx < 0 || nx > end || ny < 0 || ny > end) {
                continue;
            }
            if (grid[nx][ny] == cur + 1) {
                dfs(nx, ny, cur + 1, end, grid);
            }
        }
    }

    public int[] prevPermOpt1(int[] arr) {
        int n = arr.length;
        for (int i = n - 2;i >= 0; i--) {
            if (arr[i] > arr[i + 1]) {
                int j = n - 1;
                while (arr[j] >= arr[i]) {
                    j--;
                }
                int t = arr[j];
                arr[j] = arr[i];
                arr[i] = t;
                break;
            }
        }
        return arr;
    }

    int ans44 = -1;

    public int beautifulSubsets(int[] nums, int k) {
        int n = nums.length;
        int[] vis = new int[5050];
        dfs(0, nums, vis, k);
        return ans44;
    }

    public void dfs(int cur, int[] nums, int[] vis, int k) {
        ans44++;
        if (cur == nums.length) {
            return;
        }
        for (int j = cur;j < nums.length; j++) {
            int c = nums[j] + k;
            if (vis[c - k] == 0 && vis[c + k] == 0) {
                vis[c]++;
                dfs(j + 1, nums, vis, k);
                vis[c]--;
            }
        }
    }

    public int findSmallestInteger(int[] nums, int m) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int x : nums) {
            map.merge((x % m + m) % m, 1, Integer::sum);
        }
        int ans = 0;
        while (map.merge(ans % m, -1, Integer::sum) >= 0) {
            ans++;
        }
        return ans;
    }

    public String baseNeg2(int n) {
        StringBuilder sb = new StringBuilder();
        while (n > 0) {
            sb.append(n % 2);
            n >>= 1;
        }
        if ((n & (n - 1)) != 0) {
            sb.append("1");
        }
        return sb.reverse().toString();
    }

    public int[] supplyWagon(int[] supplies) {
        int n = supplies.length;
        int c = n >> 1;
        List<Integer> list = new ArrayList<>();
        for (int i = 0;i < n; i++) {
            list.add(supplies[i]);
        }
        while (list.size() > c) {
            int vv = list.size();
            int cc = 1;
            int min = Integer.MAX_VALUE;
            for (int i = 1;i < vv; i++) {
                int cur = list.get(i) + list.get(i - 1);
                if (cur < min) {
                    cc = i;
                    min = cur;
                }
            }
            list.set(cc, min);
            list.remove(cc - 1);
        }
        int[] ans = new int[list.size()];
        for (int i = 0;i < list.size(); i++) {
            ans[i] = list.get(i);
        }
        return ans;
    }

    public int adventureCamp(String[] expeditions) {
        int ans = -1;
        String start = expeditions[0];
        String[] strings = start.split("->");
        Set<String> set = Arrays.stream(strings).collect(Collectors.toSet());
        int max = 0;
        for (int i = 1;i < expeditions.length; i++) {
            String cur = expeditions[i];
            String[] curs = cur.split("->");
            Set<String> set1 = new HashSet<>();
            for (int j = 0;j < curs.length; j++) {
                if (curs[j].length() > 0 && !set.contains(curs[j])) {
                    set1.add(curs[j]);
                }
            }
            set.addAll(set1);
            if (set1.size() > max) {
                max = set1.size();
                ans = i;
            }
        }
        return ans;
    }

    public int fieldOfGreatestBlessing(int[][] mat) {
        int ans = 1;
        TreeMap<Double, List<Integer>> map = new TreeMap<>();
        Map<Integer, double[]> map1 = new HashMap<>();
        for (int i = 0;i < mat.length; i++) {
            double cur = mat[i][2] / 2.0;
            double y1 = mat[i][1] - cur;
            double y2 = mat[i][1] + cur;
            double x1 = mat[i][0] - cur;
            double x2 = mat[i][0] + cur;
            map.putIfAbsent(y1, new ArrayList<>());
            map.putIfAbsent(y2, new ArrayList<>());
            map.get(y1).add(i);
            map.get(y2).add(i);
            map1.put(i, new double[]{x1, x2});
        }
        Set<Integer> set = new HashSet<>();
        for (Double key : map.keySet()) {
            List<Integer> needRemove = new ArrayList<>();
            for (int index : map.get(key)) {
                if (!set.contains(index)) {
                    set.add(index);
                } else {
                    needRemove.add(index);
                }
            }
            List<double[]> list = new ArrayList<>();
            for (int x : set) {
                list.add(map1.get(x));
            }
            ans = Math.max(ans, maxOverlap(list));
            for (int x : needRemove) {
                set.remove(x);
            }
        }
        return ans;
    }

    public int maxOverlap(List<double[]> intervals) {
        List<double[]> events = new ArrayList<>();
        for (double[] interval : intervals) {
            events.add(new double[]{interval[0], 1});
            events.add(new double[]{interval[1], -1});
        }

        events.sort((o1, o2) -> {
            if (o1[0] != o2[0]) {
                return Double.compare(o1[0], o2[0]);
            }
            return Double.compare(o2[1], o1[1]);
        });

        int count = 0;
        int maxCount = 0;
        for (double[] event : events) {
            count += event[1];
            maxCount = Math.max(maxCount, count);
        }

        return maxCount;
    }

    public int rampartDefensiveLine(int[][] rampart) {
        int l = 0, r = (int) (6);
        int ans = 0;
        while (l <= r) {
            int mid = (l + r) >> 1;
            if (check(rampart, mid)) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }

    public boolean check(int[][] rampart, int mid) {
        int last = rampart[0][1];
        for (int i = 1;i < rampart.length - 1; i++) {
            int s = rampart[i][0];
            int e = rampart[i][1];
            int m1 = Math.max(last, rampart[i - 1][1]);
            int m2 = rampart[i + 1][0];
            int dis1 = s - m1;
            int dis2 = m2 - e;
            if (dis1 + dis2 < mid) {
                return false;
            }
            if (mid >= dis1) {
                last = e + (mid - dis1);
            } else {
                last = e;
            }
        }
        return true;
    }

    int[][] dirs4 = {
            {1, 0}, {0, 1},
            {-1, 0}, {0, -1}
    };

    int[] vis = new int[1638401];

    public int extractMantra(String[] matrix, String mantra) {
        int n = matrix.length;
        int m = matrix[0].length();
        Queue<Integer> queue = new ArrayDeque<>();
        int v = tuple3ToLong(0, 0, 0);
        queue.add(v);
        vis[v] = 1;
        int step = 1;
        while (!queue.isEmpty()) {
            for (int i = queue.size() - 1;i >= 0; i--) {
                int cur = queue.poll();
                int v3 = cur & ((1 << 7) - 1);
                int v2 = (cur >> 7) & ((1 << 7) - 1);
                int v1 = cur >> 14;
                if (matrix[v1].charAt(v2) == mantra.charAt(v3)) {
                    if (v3 == mantra.length() - 1) {
                        return step;
                    }
                    int hash = tuple3ToLong(v1, v2, v3 + 1);
                    if (vis[hash] == 0) {
                        queue.add(hash);
                        vis[hash] = 1;
                    }
                }
                for (int[] dir : dirs4) {
                    int x = v1 + dir[0];
                    int y = v2 + dir[1];
                    int hash = tuple3ToLong(x, y, v3);
                    if (x >= 0 && x < n && y >= 0 && y < m && vis[hash] == 0) {
                        vis[hash] = 1;
                        queue.add(hash);
                    }
                }
            }
            step++;
        }
        return -1;
    }

    public int tuple3ToLong(int a, int b, int c) {
        return a << 14 | b << 7 | c;
    }

    public String evolutionaryRecord(int[] parents) {
        int n = parents.length;
        List<Integer>[] lists = new ArrayList[n];
        for (int i = 0;i < n; i++) {
            lists[i] = new ArrayList<>();
        }
        for (int i = 1;i < parents.length; i++) {
            int p = parents[i];
            lists[p].add(i);
        }
        String res = dfs(0, lists);
        int end = res.length() - 1;
        while (res.charAt(end) == '1') {
            end--;
        }
        return res.substring(1, end + 1);
    }

    public String dfs(int index, List<Integer>[] lists) {
        if (lists[index].size() == 0) {
            return "01";
        }
        Set<String> list = new TreeSet<>();
        for (int x : lists[index]) {
            list.add(dfs(x, lists));
        }
        return "0" + String.join("", list) + "1";
    }

    public int[] circularGameLosers(int n, int k) {
        int[] cur = new int[n];
        List<Integer> res = new ArrayList<>();
        int i = 0, v = 1;
        while (true) {
            if (cur[i] == 1) {
                break;
            }
            cur[i] = 1;
            i += (v * k);
            if (i >= n) {
                i %= n;
            }
        }
        for (int vv = 0;vv <n; vv++) {
            if (cur[vv] == 0) {
                res.add(vv + 1);
            }
        }
        int[] ans = new int[res.size()];
        for (int vv = 0;vv < ans.length; vv++) {
            ans[vv] = res.get(vv);
        }
        return ans;
    }

    public boolean doesValidArrayExist(int[] derived) {
        return judge(0, derived) || judge(1, derived);
    }

    private boolean judge(int s, int[] derived) {
        int cur = s;
        int next = 1;
        for (int i = 0;i < derived.length; i++) {
            int vv = derived[i];
            if (i != derived.length - 1) {
                if (vv == 1) {
                    if (cur == 1) {
                        next = 0;
                    } else {
                        next = 1;
                    }
                } else if (vv == 0) {
                    if (cur == 1) {
                        next = 1;
                    } else {
                        next = 0;
                    }
                }
            } else {
                break;
            }
            cur = next;
        }
        return (cur ^ s) == derived[derived.length - 1];
    }

    int[][] dirs3 = {
            {1, 1}, {0, 1},
            {-1, 1}
    };

    public int maxMoves(int[][] grid) {
        int ans = 0;
        int n = grid.length;
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0;i < n; i++) {
            ans = Math.max(ans, maxMoves1(grid, i, 0, map));
        }
        return ans;
    }

    public int maxMoves1(int[][] grid, int s, int e, Map<String, Integer> map) {
        if (map.get(s + "," + e) != null) {
            return map.get(s + "," + e);
        }
        int t = 0;
        for (int[] dir : dirs3) {
            int nx = dir[0] + s;
            int ny = dir[1] + e;
            if (nx >= 0 && nx < grid.length && ny >= 0 && ny < grid[0].length) {
                if (grid[nx][ny] > grid[s][e]) {
                    t = Math.max(t, maxMoves1(grid, nx, ny, map));
                }
            }
        }
        map.put(s + "," + e, t);
        return t;
    }

    int[] vis1 = null;

    public int countCompleteComponents(int n, int[][] edges) {
        List<Integer>[] lists = new ArrayList[n];
        vis1 = new int[n];
        for (int i = 0;i < n; i++) {
            lists[i] = new ArrayList<>();
        }
        for (int[] e : edges) {
            lists[e[0]].add(e[1]);
            lists[e[1]].add(e[0]);
        }
        int ans = 0;
        for (int i = 0;i < n; i++) {
            if (vis1[i] == 0) {
                List<Integer> cur = new ArrayList<>();
                dfs1(i, lists, cur);
                boolean flag = true;
                for (int x : cur) {
                    List<Integer> vv = lists[x];
                    Set<Integer> set = new HashSet<>(cur);
                    set.remove(x);
                    if (set.size() != vv.size()) {
                        flag = false;
                    }
                    if (!flag) {
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

    public void dfs1(int i, List<Integer>[] lists, List<Integer> list) {
        list.add(i);
        vis1[i] = 1;
        for (int x : lists[i]) {
            if (vis1[x] == 0) {
                dfs1(x, lists, list);
            }
        }
    }

    public int countSeniors(String[] details) {
        int ans = 0;
        for (String d : details) {
            String cur = "" +  d.charAt(10) + d.charAt(11);
            if (Integer.parseInt(cur) > 60) {
                ans++;
            }
        }
        return ans;
    }

    public int matrixSum(int[][] nums) {
        for (int[] num : nums) {
            Arrays.sort(num);
        }
        int m = nums[0].length - 1;
        int ans = 0;
        for (int j = m;j >= 0; j--) {
            int vv = 0;
            for (int i = 0;i < nums.length; i++) {
                vv = Math.max(vv, nums[i][j]);
            }
            ans += vv;
        }
        return ans;
    }

    public long maximumOr(int[] nums, int k) {
        int pre = 0;
        int n = nums.length;
        int[] suf = new int[n + 1];
        for (int i = n - 1;i >= 0; i--) {
            suf[i] = suf[i + 1] | nums[i];
        }
        int ans = 0;
        for (int i = 0;i < n; i++) {
            ans = Math.max(ans, pre | (nums[i] << k) | suf[i + 1]);
            pre |= nums[i];
        }
        return ans;
    }

    public int sumOfPower(int[] nums) {
        Arrays.sort(nums);
        int mod = (int) (1e9 + 7);
        long ans = 0, sum = 0;
        for (int num : nums) {
            ans += ((long)num * num % mod) * (num + sum) % mod;
            sum = (sum * 2 + num) % mod;
        }
        return (int) (ans % mod);
    }

    public int[] rearrangeBarcodes(int[] barcodes) {
        Arrays.sort(barcodes);
        int[] ans = new int[barcodes.length];
        int l = 0, r = barcodes.length - 1;
        for (int i = 0;i < barcodes.length; i+=2) {
            ans[i] = barcodes[l++];
            ans[i] = barcodes[r--];
        }
        if (l == r) {
            ans[ans.length - 1] = barcodes[l];
        }
        return ans;
    }

    public int minLength(String s) {
        while (s.contains("AB") || s.contains("CD")) {
            s = s.replace("AB", "");
        }
        return s.length();
    }

    public String makeSmallestPalindrome(String s) {
        char[] chars = s.toCharArray();
        int n = s.length();
        int ans = 0;
        for (int i = 0;i < n / 2; i++) {
            char x = chars[i];
            char y = chars[n - i - 1];
            if (x != y) {
                if (x > y) {
                    chars[i] = y;
                } else {
                    chars[n - i - 1] = x;
                }
            }
        }
        return new String(chars);
    }

//    public int punishmentNumber(int n) {
//
//    }

    boolean flag = false;

    public void dfs(String str, int s, int cur, int n) {
        if (s == str.length()) {
            if (cur == n) {
                flag = true;
            }
            return;
        }
        for (int i = s;i < str.length(); i++) {
            String vv = str.substring(s, i + 1);
            dfs(str, i + 1, cur + Integer.parseInt(vv), n);
        }
    }
    public int punishmentNumber(int n) {
        int ans = 0;
        for (int i = 1;i <= n; i++) {
            dfs(String.valueOf(i * i), 0, 0, i);
            if (flag) {
                ans += i * i;
            }
            flag = false;
        }
        return ans;
    }

    public int buyChoco(int[] prices, int money) {
        Arrays.sort(prices);
        return prices[0] + prices[1] > money ? money : money - (prices[0] + prices[1]);
    }

    public int minExtraCharWithSearch(String s, String[] dictionary) {
        Set<String> set = Arrays.stream(dictionary).collect(Collectors.toSet());
        Map<Integer, Integer> map = new HashMap<>();
        Function<Integer, Integer> dfs = new Function<Integer, Integer>() {
            @Override
            public Integer apply(Integer i) {
                if (map.get(i) != null) {
                    return map.get(i);
                }
                if (i < 0) {
                    return 0;
                }
                int res = apply(i - 1) + 1;
                for (int j = 0;j < i + 1; j++) {
                    if (set.contains(s.substring(j, i + 1))) {
                        res = Math.min(res, apply(j - 1));
                    }
                }
                map.put(i, res);
                return res;
            }
        };
        return dfs.apply(s.length() - 1);
    }

    public int minExtraChar(String s, String[] dictionary) {
        Set<String> set = Arrays.stream(dictionary).collect(Collectors.toSet());
        int n = s.length();
        int[] f = new int[n + 1];
        f[0] = 0;
        for (int i = 0;i < n; i++) {
            f[i + 1] = f[i] + 1;
            for (int j = 0;j <= i; j++) {
                String sub = s.substring(j, i + 1);
                if (set.contains(sub)) {
                    f[i + 1] = Math.min(f[i + 1], f[j]);
                }
            }
        }
        return f[n];
    }

    public long maxStrengthWith2N(int[] nums) {
        int n = nums.length;
        long ans = Integer.MIN_VALUE;
        for (int i = 1;i < (1 << n); i++) {
            int cur = i;
            int index = 0;
            long vv = 1;
            while (cur > 0) {
                if ((cur & 1) == 1) {
                    vv *= nums[index];
                }
                index++;
                cur >>= 1;
            }
            ans = Math.max(ans, vv);
        }
        return ans;
    }

    long ans0604 = Long.MIN_VALUE;

    public long maxStrength(int[] nums) {
        dfs(0, 1, true, nums);
        return ans0604;
    }

    public void dfs(int i, long prod, boolean empty, int[] nums) {
        if (i == nums.length) {
            if (!empty) {
                ans0604 = Math.max(ans0604, prod);
            }
            return;
        }
        dfs(i + 1, prod * nums[i], false, nums);
        dfs(i + 1, prod, empty, nums);
    }

    public long maxStrengthDP(int[] nums) {
        long min = nums[0], max = nums[0];
        for (int i = 1;i < nums.length; i++) {
            long tmp = max;
            max = max(max, (long) nums[i], max * nums[i], min * nums[i]);
            min = min(min, (long) nums[i], tmp * nums[i], min * nums[i]);
        }
        return max;
    }

    public Long max(Long... integers) {
        return Arrays.stream(integers).max(Long::compare).get();
    }

    public Long min(Long... integers) {
        return Arrays.stream(integers).min(Long::compare).get();
    }

    public int[][] differenceOfDistinctValues(int[][] grid) {
        int n = grid.length;
        int m = grid[0].length;
        int[][] ans = new int[n][m];
        for (int i = 0;i < n; i++) {
            for (int j = 0;j < m; j++) {
                Set<Integer> set1 = new HashSet<>();
                Set<Integer> set2 = new HashSet<>();
                int c1 = i - 1, c2 = j - 1, c3 = i + 1, c4 = j + 1;
                while (c1 >= 0 && c2 >= 0) {
                    set1.add(grid[c1][c2]);
                    c1--;c2--;
                }
                while (c3 < n && c4 < m) {
                    set2.add(grid[c3][c4]);
                    c3++;c4++;
                }
                ans[i][j] = Math.abs(set1.size() - set2.size());
            }
        }
        return ans;
    }

    public long minCost(int[] nums, int x) {
        int n = nums.length;
        long[] sum = new long[n];
        for (int i = 0;i < n; i++) {
            sum[i] = (long) i * x;
        }
        for (int i = 0;i < n; i++) {
            int min = nums[i];
            for (int j = i;j < i + n; j++) {
                min = Math.min(min, nums[j]);
                sum[j - i] += min;
            }
        }
        return Arrays.stream(sum).min().getAsLong();
    }

    public int findNonMinOrMax(int[] nums) {
        int max = Arrays.stream(nums).max().getAsInt();
        int min = Arrays.stream(nums).min().getAsInt();
        return Arrays.stream(nums).filter(x -> x != max && x != min).min().orElse(-1);
    }

    public String smallestString(String s) {
        int n = s.length();
        StringBuilder sb = new StringBuilder();
        int v = 0;
        while (v < n && s.charAt(v) == 'a') {
            if (v == n - 1) {
                sb.append('z');
                break;
            }
            sb.append('a');
            v++;
        }
        for (int i = v;i < n; i++) {
            char c = s.charAt(i);
            if (c == 'a' && i != 0) {
                sb.append(s.substring(i));
                break;
            }
            sb.append((char) ('a' + (((c - 'a') + 25)) % 26));
        }
        return sb.toString();
    }

    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode hair = new ListNode(0);
        ListNode p = hair;
        while (list1 != null && list2 != null) {
            if (list1.val > list2.val) {
                hair.next = list2;
                list2 = list2.next;
            } else {
                hair.next = list1;
                list1 = list1.next;
            }
            hair = hair.next;
        }
        if (list1 == null) {
            hair.next = list2;
        } else {
            hair.next = list1;
        }
        return p.next;
    }

    public static void main(String[] args) {
        int haw = new NewNewMain().minExtraChar("haw", new String[]{"aw"});
        System.out.println(haw);
    }
}
