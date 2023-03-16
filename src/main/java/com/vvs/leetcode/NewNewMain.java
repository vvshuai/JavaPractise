package com.vvs.leetcode;

import java.util.*;
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

    public static void main(String[] args) {
        new NewNewMain()
                .minimumEffort(new int[][]{
                        {1,2},{2,4},{4,8}
                });
    }
}
