package com.vvs.controller;

import com.vvs.multithread.Val;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 23:06 2020/9/1
 * @Modified By:
 */
@RestController
public class StatController {

    static HashSet<Val<Integer>> set = new HashSet<>();

    synchronized static void addSet(Val<Integer> v){
        set.add(v);
    }

    static ThreadLocal<Val<Integer>> c = new ThreadLocal<Val<Integer>>(){
        @Override
        protected Val<Integer> initialValue(){
            Val<Integer> v = new Val<>();
            v.set(0);
            addSet(v);
            return v;
        }
    };

    void __add() throws InterruptedException {
        Thread.sleep(100);
        Val<Integer> v = c.get();
        v.set(v.get()+1);
    }

    @RequestMapping("/stat")
    public Integer stat(){
        return set.stream().map(x -> x.get()).reduce((a, x) -> x+a).get();
    }


    @RequestMapping("/add")
    public Integer add() throws InterruptedException {
        __add();
        return 1;
    }

    @RequestMapping("/retry/DispatchToInnerService")
    @ResponseBody
    public void retryDispatchToInnerService() {
        // do something...

    }
    @RequestMapping("/retry/postTaskResultToRedis")
    @ResponseBody
    public void retryPostTaskResultToRedis() {
        // do something...

    }

    @RequestMapping("/recoverdata/picidlist")
    @ResponseBody
    public void recoverRenderDataWithPicIdList(final List<Long> picIdList) {
        // do something...

    }

    @RequestMapping("/recoverdata/taskidlist")
    @ResponseBody
    public void recoverRenderDataWithTaskIdList(final List<String> taskIdList) {
        // do something...

    }
}
