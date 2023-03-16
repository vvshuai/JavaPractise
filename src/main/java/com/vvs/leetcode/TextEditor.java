package com.vvs.leetcode;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 20:16 2022/6/5
 * @Modified By:
 */
class TextEditor {

    StringBuilder sb;
    int cur = 0;

    public TextEditor() {
        sb = new StringBuilder();
        sb.append("|");
    }

    public void addText(String text) {
        sb.insert(cur, text);
        cur += text.length();
    }

    public int deleteText(int k) {
        int vv = k > cur ? 0 : cur - k;
        sb.delete(vv, cur);
        int ans = cur - vv;
        cur -= vv;
        System.out.println(sb.toString());
        return ans;
    }

    public String cursorLeft(int k) {
        int vv = k > cur ? 0 : cur - k;
        sb.deleteCharAt(cur);
        sb.insert(vv, "|");
        cur = vv;
        return sb.substring(Math.max(vv - 10, 0), vv);
    }

    public String cursorRight(int k) {
        int vv = cur + k >= sb.length() ? sb.length() - 1 : cur + k;
        sb.deleteCharAt(cur);
        sb.insert(vv, "|");
        cur = vv;
        return sb.substring(Math.max(vv - 10, 0), vv);
    }
}
