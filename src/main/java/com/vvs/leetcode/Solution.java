package com.vvs.leetcode;

import javax.sound.sampled.Line;
import java.util.*;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 15:58 2020/6/11
 * @Modified By:
 * */
class Solution {

    Map<Integer, List<Integer>> map;
    Random random = new Random();
    public Solution(int[] nums) {
        map = new HashMap<>();
        for (int i = 0;i < nums.length; i++) {
            map.putIfAbsent(nums[i], new ArrayList<>());
            map.get(nums[i]).add(i);
        }
    }

    public int pick(int target) {
        List<Integer> list = map.get(target);
        return list.get(random.nextInt(list.size()));
    }

    public static int findLatestStep(int[] arr, int m) {
        int n = arr.length;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i <= n; i++) {
            sb.append('0');
        }
        int ans = 0;
        int Min = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            sb.setCharAt(arr[i], '1');
            if (i < m - 1) {
                continue;
            }
            String[] strings = sb.toString().split("0");
            for (String s : strings) {
                if (s.length() == m) {
                    ans = i + 1;
                }
                Min = Math.min(Min, s.length());
            }
            if (Min > m) {
                break;
            }
        }
        return ans == 0 ? -1 : ans;
    }

    public boolean hasCycle(ListNode head){
        if(head == null || head.next == null){
            return false;
        }
        ListNode slow = head;
        ListNode fast = head.next;
        while(slow != fast){
            if(fast == null || fast.next == null){
                break;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }

    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        int n = rooms.size();
        boolean[] vis = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (vis[i]) {
                dfs(vis, i, rooms);
            }
        }
        for (int i = 0; i < n; i++) {
            if (vis[i] == false) {
                return false;
            }
        }
        return true;
    }

    private void dfs(boolean[] vis, int x, List<List<Integer>> rooms) {
        if (vis[x] == true) {
            return;
        }
        vis[x] = true;
        List<Integer> list = rooms.get(x);
        for (int i = 0; i < list.size(); i++) {
            dfs(vis, list.get(i), rooms);
        }
    }


    public int[] findMaxRightWithStack(int[] array) {
        if (array == null) {
            return array;
        }
        int n = array.length;
        int[] result = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(0);
        int index = 1;
        while (index < n) {
            if (!stack.isEmpty() && result[index] > result[stack.peek()]) {
                result[stack.poll()] = result[index];
            } else {
                stack.push(index);
                index++;
            }
        }
        while (!stack.isEmpty()) {
            result[stack.pop()] = -1;
        }
        return result;
    }

    public static List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> map = new HashMap<>();
        for (String s : words) {
            map.put(s, map.getOrDefault(s, 0) + 1);
        }
        Queue<String> que = new PriorityQueue<>();
        Pair[] pairs = new Pair[map.size()];
        int i = 0;
        for (String s : map.keySet()) {
            pairs[i] = new Pair(s, map.get(s));
            i++;
        }
        Arrays.sort(pairs, new Comparator<Pair>() {
            @Override
            public int compare(Pair o1, Pair o2) {
                if (o2.getSum() != o1.getSum()) {
                    return o2.getSum() - o1.getSum();
                }
                return o2.getS().compareTo(o1.getS());
            }
        });
        List<String> ans = new ArrayList<>();
        for (int j = 0; j < pairs.length; i++) {
            ans.add(pairs[i].getS());
        }
        return ans;
    }

    static class Pair {
        private String s;
        private int sum;

        public Pair(String s, int sum) {
            this.s = s;
            this.sum = sum;
        }

        public Pair() {
        }

        public String getS() {
            return s;
        }

        public void setS(String s) {
            this.s = s;
        }

        public int getSum() {
            return sum;
        }

        public void setSum(int sum) {
            this.sum = sum;
        }
    }

    public List<String> binaryTreePaths(TreeNode root) {
        List<String> ans = new ArrayList<>();
        dfs(root, "", ans);
        return ans;
    }

    private void dfs(TreeNode root, String s, List<String> ans) {
        if (root != null) {
            StringBuilder sb = new StringBuilder(s);
            sb.append(Integer.toString(root.val));
            if (root.left == null && root.right == null) {
                ans.add(sb.toString());
            } else {
                sb.append("->");
                dfs(root.left, sb.toString(), ans);
                dfs(root.right, sb.toString(), ans);
            }
        }
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode hair = new ListNode(0);
        hair.next = head;
        ListNode pre = hair;
        ListNode end = hair;
        while (end.next != null) {
            for (int i = 0; i < k && end != null; i++) {
                end = end.next;
            }
            if (end == null) {
                break;
            }
            ListNode next = end.next;
            end.next = null;
            ListNode start = pre.next;
            pre.next = reverse(start);
            start.next = next;
            pre = start;
            end = start;
        }
        return hair.next;
    }

    public ListNode reverse(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode t = cur.next;
            cur.next = pre;
            pre = cur;
            cur = t;
        }
        return pre;
    }

    private final static int mod = 1000000007;

    public static int breakfastNumber(int[] staple, int[] drinks, int x) {
        Arrays.sort(staple);
        Arrays.sort(drinks);
        int ans = 0;
        for (int i = 0; i < staple.length; i++) {
            int vv = x - staple[i];
            if (vv < 0) {
                continue;
            } else {
                int index = search(drinks, drinks.length-1, vv);
                if(index > 0){
                    ans += index;
                    ans %= mod;
                }
            }
        }
        ans %= mod;
        return ans;
    }

    public static int search(int[] A, int n, int target) {
        int low = 0, high = n, mid;
        while (low <= high) {
            mid = low + (high - low) / 2;
            if (target >= A[mid]) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return low > n ? n+1 : low;
    }

//    public static void main(String[] args) {
//        TreeNode root = new TreeNode(1);
//        root.left = new TreeNode(2);
//        root.right = new TreeNode(3);
//        root.left.left = new TreeNode(4);
//        root.left.right = new TreeNode(5);
//        sumOfLeftLeaves(root);
//    }

    private static int[][] vis;
    private static int n, m;
    private static boolean flag;

    public static boolean exist(char[][] board, String word) {
        flag = false;
        n = board.length;
        m = board[0].length;
        for(int i = 0;i < n; i++){
            for(int j = 0;j < n; j++){
                if(board[i][j] == word.charAt(0)){
                    vis = new int[n][m];
                    dfs(board, i, j, 0, word);
                }
            }
        }
        return flag;
    }

    private static void dfs(char[][] board,int x, int y, int cur, String word){
        if(cur == word.length()){
            flag = true;
            return ;
        }
        if(x < 0 || x >= n || y < 0 || y >= m){
            return ;
        }
        if(vis[x][y] == 1){
            return ;
        }
        vis[x][y] = 1;
        if(board[x][y] == word.charAt(cur)){
            dfs(board, x-1, y, cur+1, word);
            dfs(board, x+1, y, cur+1, word);
            dfs(board, x, y-1, cur+1, word);
            dfs(board, x, y+1, cur+1, word);
        }
    }

    private static int ans = 0;

    public static int sumOfLeftLeaves(TreeNode root) {
        if(root == null){
            return 0;
        }
        if(root.left != null){
            ans += root.left.val;
        }
        sum(root.left);
        sum(root.right);
        return ans;
    }
    public static void sum(TreeNode root){
        if(root == null){
            return ;
        }
        if(root.left != null){
            ans += root.left.val;
        }
        sum(root.left);
        sum(root.right);
    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> list = new ArrayList<>();
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        int cur = 0;
        while(!que.isEmpty()){
            List<Integer> l = new ArrayList<>();
            for(int i = que.size()-1;i >= 0; i--){
                TreeNode t = que.poll();
                l.add(t.val);
                if(t.left != null){
                    que.add(t.left);
                }
                if(t.right != null){
                    que.add(t.right);
                }
            }
            cur++;
            if(cur %2 == 0){
                Collections.reverse(l);
            }
            list.add(l);
        }
        return list;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        return merge(lists, 0, lists.length-1);
    }

    private ListNode merge(ListNode[] lists, int l, int r) {
        if(l == r){
            return lists[l];
        }
        if(l > r){
            return null;
        }
        int mid = (l + r) >> 1;
        return mergeTwoList(merge(lists, l, mid), merge(lists, mid+1, r));
    }

    private ListNode mergeTwoList(ListNode h1, ListNode h2) {
        if(h1 == null){
            return h2;
        }
        if(h2 == null){
            return h1;
        }
        ListNode head = new ListNode(0);
        ListNode p = head;
        while(h1 != null && h2 != null){
            if(h1.val > h2.val){
                p.next = h2;
                h2 = h2.next;
            } else {
                p.next = h1;
                h1 = h1.next;
            }
            p = p.next;
        }
        p.next = h1==null ? h2 : h1;
        return head.next;
    }

    public int lengthOfLongestSubstring(String s) {
        if(s.length() == 0){
            return 0;
        }
        Map<Character, Integer> map = new HashMap<>();
        int l = 0;
        int ans = 0;
        for(int i = 0;i < s.length(); i++){
            if(map.containsKey(s.charAt(i))){
                l = Math.max(l, map.get(s.charAt(i))+1);
            }
            ans = Math.max(ans, i-l+1);
            map.put(s.charAt(i), i);
        }
        return ans;
    }


    public ListNode reverseList(ListNode head) {
        if(head.next == null || head == null){
            return head;
        }
        ListNode ret = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return ret;
    }

    public int maxProfit(int[] prices){
        if(prices.length == 0){
            return 0;
        }
        int ans = 0;
        int min = prices[0];
        for(int i = 1;i < prices.length; i++){
            min = Math.min(prices[i], min);
            ans = Math.max(ans, prices[i] - min);
        }
        return ans;
    }

    public Node connect(Node root) {
        Queue<Node> que = new LinkedList<>();
        if(root != null){
            que.add(root);
        }
        while(que.isEmpty()){
            int cur = que.size();
            Node pre = que.poll();
            if(pre.left != null){
                que.add(pre.left);
            }
            if(pre.right != null){
                que.add(pre.right);
            }
            for(int i = cur-2;i >= 0; i--){
                Node node = que.poll();
                pre.next = node;
                pre = node;
            }
        }
        return root;
    }

    public int trap(int[] height) {
        int ans = 0;
        Deque<Integer> stack = new LinkedList<>();
        int n = height.length;

        for(int i = 0;i < n; i++) {
            while(!stack.isEmpty() && height[stack.peek()] < height[i]) {
                int top = stack.pop();
                if(stack.isEmpty()) {
                    break;
                }
                int l = stack.peek();
                int wide = i-l-1;
                int high = Math.min(height[l], height[i]) - height[top];
                ans += wide*high;
            }
            stack.push(i);
        }
        return ans;
    }

    public int longestCommonSubsequence(String text1, String text2) {
        int n = text1.length();
        int m = text2.length();
        int[][] dp = new int[n+1][m+1];
        for(int i = 1;i <= n; i++) {
            for(int j = 1;j <= m; j++) {
                char x = text1.charAt(i);
                char y = text2.charAt(j);

                if(x != y) {
                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
                } else {
                    dp[i][j] = dp[i-1][j-1] + 1;
                }
            }
        }

        return dp[n][m];
    }

    public int removeDuplicates(int[] nums) {
        int length = nums.length;
        if(length <= 2) {
            return length;
        }
        int i = 2;
        for(int j = 2;j < length; j++) {
            if(nums[i-2] != nums[j]) {
                nums[i++] = nums[j];
            }
        }
        return i;
    }

    private static int vvs = 0;

    public static int numDecodings(String s) {
        int n = s.length();
        int[] arr = new int[n+1];
        arr[0] = 1;
        for(int i = 1;i <= n; i++) {
            if(s.charAt(i-1)  != '0') {
                arr[i] += arr[i-1];
            }
            if(i > 1 && s.charAt(i-2) != '0' &&  ((s.charAt(i-2)-'0')*10 + (s.charAt(i-1)-'0')) <= 26) {
                arr[i] += arr[i-2];
            }
        }
        return arr[n];
    }

    public static void dfs(String str, int s, int e) {
        if(e == str.length()) {
            if(s == e) {
                vvs++;
            }
            return ;
        }
        if(e > str.length()) {
            return;
        }
        if(s == e) {
            int cur = str.charAt(s) - '0';
            if(cur <= 0 || cur > 9) {
                return;
            } else {
                dfs(str, e+1, e+1);
                dfs(str, e+1, e+2);
            }
        } else {
            int cur = (str.charAt(s) - '0') * 10 + (str.charAt(e) - '0');
            if(cur < 10 || cur > 26) {
                return ;
            } else {
                dfs(str, e+1, e+1);
                dfs(str, e+1, e+2);
            }
        }
    }

    public static void main(String[] args) {
        long startTime =  System.currentTimeMillis();
        int i = numDecodings("18110");

        System.out.println(i);
        System.out.println(1836311903);
    }
}
class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
};