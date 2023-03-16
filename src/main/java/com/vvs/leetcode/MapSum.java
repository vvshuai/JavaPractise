package com.vvs.leetcode;

import java.util.HashMap;
import java.util.Map;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 11:27 2021/11/14
 * @Modified By:
 */
class MapSum {

    private Map<String, Integer> map;
    private Node root;
    class Node{
        int val = 0;
        Node[] next = new Node[26];
    }

    public MapSum() {
        map = new HashMap<>();
        root = new Node();
    }

    public void insert(String key, int val) {
        map.put(key, val);
        int cur = val - map.getOrDefault(key, 0);
        Node node = root;
        for (char c : key.toCharArray()) {
            if (node.next[c-'a'] == null) {
                node.next[c-'a'] = new Node();
            }
            node = node.next[c-'a'];
            node.val += cur;
        }
    }

    public int sum(String prefix) {
        Node node = root;
        for (char c : prefix.toCharArray()) {
            if (node.next[c-'a'] == null) {
                return 0;
            }
            node = node.next[c-'a'];
        }
        return node.val;
    }

    public static void main(String[] args) {
        String s = "ESPESTCOPIPCNTDPYPPODACZRCLXXTYR".toLowerCase();
        for (int i = 0;i < 26; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0;j < s.length(); j++) {
                char cur = s.charAt(j);
                sb.append((char)((cur - 'a' - i + 26) % 26 + 'a'));
            }
            System.out.println(sb);
        }
    }
}
