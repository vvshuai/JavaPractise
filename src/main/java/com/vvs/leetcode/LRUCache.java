package com.vvs.leetcode;

import java.util.HashMap;
import java.util.Map;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 21:00 2020/7/30
 * @Modified By:
 */
public class LRUCache {

    class DLinkNode{
        int key;
        int value;
        DLinkNode prev;
        DLinkNode next;
        public DLinkNode(){}
        public DLinkNode(int _key, int _value){
            key = _key;
            value = _value;
        }
    }

    private Map<Integer, DLinkNode> cache = new HashMap<>();
    private int size;
    private int capacity;
    private DLinkNode head, tail;

    public LRUCache(int capacity) {
        this.size = size;
        this.capacity = capacity;
        head = new DLinkNode();
        tail = new DLinkNode();
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        DLinkNode node = cache.get(key);
        if(node == null){
            return -1;
        }
        moveTohead(node);
        return node.value;
    }

    private void moveTohead(DLinkNode node) {
        removeNode(node);
        addToHead(node);
    }

    private void addToHead(DLinkNode node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }

    private void removeNode(DLinkNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    public void put(int key, int value) {
        DLinkNode node = cache.get(key);
        if(node == null){
            DLinkNode newNode = new DLinkNode(key, value);
            cache.put(key, newNode);
            addToHead(newNode);
            ++size;
            if(size > capacity){
                DLinkNode tail = removeTail();
                cache.remove(tail.key);
                --size;
            }
        }else{
            node.value = value;
            moveTohead(node);
        }
    }

    private DLinkNode removeTail() {
        DLinkNode res = tail.prev;
        removeNode(res);
        return res;
    }
}