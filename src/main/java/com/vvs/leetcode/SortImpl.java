package com.vvs.leetcode;

import org.apache.logging.log4j.message.Message;
import org.apache.rocketmq.client.consumer.DefaultMQPushConsumer;
import org.apache.rocketmq.client.consumer.listener.ConsumeConcurrentlyContext;
import org.apache.rocketmq.client.consumer.listener.ConsumeConcurrentlyStatus;
import org.apache.rocketmq.client.consumer.listener.MessageListenerConcurrently;
import org.apache.rocketmq.common.message.MessageExt;

import java.util.ArrayList;
import java.util.List;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 14:24 2020/8/1
 * @Modified By:
 */
public class SortImpl {

    public static void main(String[] args){
        int[] arr = {1,4,6,7,4,3,5};
        quickSort(0, arr.length-1, arr);
        for(int i = 0;i < arr.length; i++){
            System.out.print(arr[i] + " ");
        }
        DefaultMQPushConsumer mqPushConsumer = new DefaultMQPushConsumer();

    }

    public static void quickSort(int l, int r, int[] arr){
        if(l >= r){
            return ;
        }
        int low = l;
        int high = r;
        int p = arr[l];
        while(l != r){
            while(arr[r] > p && l < r) r--;
            while(arr[l] <= p && l < r) l++;
            if(l < r){
                swap(l, r, arr);
            }
        }
        swap(l, low, arr);
        quickSort(low , l-1, arr);
        quickSort(l+1, high, arr);
    }

    private static void swap(int l, int r, int[] arr) {
        int t = arr[l];
        arr[l] = arr[r];
        arr[r] = t;
    }

    public static void quicksort1(int l, int r, int[] arr){
        if(l >= r){
            return ;
        }
        int low = l;
        int high = r;
        int p = arr[l];
        while(l < r){
            while(l < r && arr[r] > p) r--;
            arr[l] = arr[r];
            while(l < r && arr[l] <= p) l++;
            arr[r] = arr[l];
        }
        arr[l] = p;
        quicksort1(low, l-1, arr);
        quicksort1(l+1, high, arr);
    }

    public static void quicksort(int l, int r, int[] arr){
        if(l <= r){
            return ;
        }
        int low = l;
        int high = r;
        int p = arr[l];
        while(l < r){
            while(l < r && arr[r] > p) r--;
            arr[l] = arr[r];
            while(l < r && arr[l] >= p) l++;
            arr[r] = arr[l];
        }
        arr[l] = p;
        quicksort(low, l-1, arr);
        quicksort(l+1, high, arr);
    }

}
