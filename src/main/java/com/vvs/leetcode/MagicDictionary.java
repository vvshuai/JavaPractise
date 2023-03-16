package com.vvs.leetcode;

import com.vvs.jvm0609.Father;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 23:32 2022/7/11
 * @Modified By:
 */
public class MagicDictionary {

    Trie root;

    public MagicDictionary() {
        root = new Trie();
    }

    public void buildDict(String[] dictionary) {
        for (String word : dictionary) {
            Trie cur = root;
            for (int i = 0;i < word.length(); i++) {
                char v = word.charAt(i);
                int idx = v - 'a';
                if (cur.child[idx] == null) {
                    cur.child[idx] = new Trie();
                }
                cur = cur.child[idx];
            }
            cur.isFinished = true;
        }
    }

    public boolean search(String searchWord) {
        return dfs(searchWord, root, 0, false);
    }

    private boolean dfs(String searchWord, Trie node, int pos, boolean modified) {
        if (pos == searchWord.length()) {
            return modified && node.isFinished;
        }
        int idx = searchWord.charAt(pos) - 'a';
        if (node.child[idx] != null) {
            if (dfs(searchWord, node.child[idx], pos + 1, modified)) {
                return true;
            }
        }
        if (!modified) {
            for (int i = 0;i < 26; i++) {
                if (node.child[i] != null) {
                    if (dfs(searchWord, node.child[i], pos + 1, true)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    public static void main(String[] args) {
        MagicDictionary magicDictionary = new MagicDictionary();
        magicDictionary.buildDict(new String[]{"hello", "leetcode"});
        magicDictionary.search("hello"); // 返回 False
        magicDictionary.search("hhllo"); // 将第二个 'h' 替换为 'e' 可以匹配 "hello" ，所以返回 True
        magicDictionary.search("hell"); // 返回 False
        magicDictionary.search("leetcoded"); // 返回 False
    }

    class Trie {
        private boolean isFinished;

        private Trie[] child;

        public Trie() {
            isFinished = false;
            child = new Trie[26];
        }
    }
}
