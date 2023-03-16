package com.vvs.leetcode;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 0:27 2022/7/7
 * @Modified By:
 */
public class NewTrie {

    public String replaceWords(List<String> dictionary, String sentence) {
        Trie trie = new Trie();
        for (String dic : dictionary) {
            Trie cur = trie;
            for (int i = 0;i < dic.length(); i++) {
                char c = dic.charAt(i);
                cur.children.putIfAbsent(c, new Trie());
                cur = cur.children.get(c);
            }
            cur.children.putIfAbsent('#', new Trie());
        }
        String[] words = sentence.split(" ");
        for (int i = 0;i < words.length; i++) {
            words[i] = findRoot(words[i], trie);
        }
        return String.join(" ", words);
    }

    private String findRoot(String word, Trie trie) {
        StringBuilder sb = new StringBuilder();
        Trie cur = trie;
        for (int i = 0;i < word.length(); i++) {
            char c = word.charAt(i);
            if (cur.children.containsKey('#')) {
                return sb.toString();
            }
            if (!cur.children.containsKey(c)) {
                return word;
            }
            sb.append(c);
            cur = cur.children.get(c);
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        Trie root = new Trie();
        Trie cur = root;
        String s = "a";
        for (int i = 0;i < s.length(); i++) {
            char c = s.charAt(i);
            cur.children.putIfAbsent(c, new Trie());
            cur = cur.children.get(c);
        }
        cur.children.put('#', new Trie());
        System.out.println(new NewTrie().findRoot(s, root));
    }

    static class Trie {
        Map<Character, Trie> children;

        public Trie() {
            children = new HashMap<>();
        }
    }
}
