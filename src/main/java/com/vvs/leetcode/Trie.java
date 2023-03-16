package com.vvs.leetcode;

import com.vvs.jvm0609.T;

import java.util.ArrayList;
import java.util.List;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 12:18 2020/7/9
 * @Modified By:
 */
public class Trie {

    class Node {
        Node[] nodes = new Node[26];
        // 当前节点经过的id
        List<Integer> ids = new ArrayList<>();
        int score;
    }

    int[] ans;



    public int[] sumPrefixScores(String[] words) {
        int len = words.length;
        Node root = new Node();
        for (int i = 0;i < len; i++) {
            Node cur = root;
            for (char ch : words[i].toCharArray()) {
                int c = ch - 'a';
                if (cur.nodes[c] == null) {
                    cur.nodes[c] = new Node();
                }
                cur = cur.nodes[c];
                ++cur.score;
            }
            cur.ids.add(i);
        }
        ans = new int[len];
        dfs(root, 0);
        return ans;
    }

    private void dfs(Node root, int sum) {
        sum += root.score;
        for (int id : root.ids) {
            ans[id] += sum;
        }
        for (Node node : root.nodes) {
            if (node != null) {
                dfs(node, sum);
            }
        }
    }

    public static void main(String[] args) {
        new Trie().sumPrefixScores(new String[]{"abc"});
    }
}