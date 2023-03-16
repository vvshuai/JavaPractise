package com.vvs.multithread;

import javafx.concurrent.Task;

import java.lang.reflect.Method;
import java.util.concurrent.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 11:23 2021/12/4
 * @Modified By:
 */
public class vvsPoolExecutor {

    private final ThreadPoolExecutor threadPoolExecutor;

    public vvsPoolExecutor() {
        threadPoolExecutor =  new ThreadPoolExecutor(2, 2, 1000,
                TimeUnit.MILLISECONDS, new SynchronousQueue<Runnable>(),
                Executors.defaultThreadFactory(),new ThreadPoolExecutor.CallerRunsPolicy());
    }

    public void addTask(Object invokeInstance, String methodName, Object... params) throws NoSuchMethodException {
        ThreadTask task = new ThreadTask(invokeInstance, params, methodName);
        this.threadPoolExecutor.submit(task);
    }

    public void shutdown() {
        threadPoolExecutor.shutdown();
        while (!threadPoolExecutor.isTerminated()) {

        }
        System.out.println("fuck!");
    }
}

class ThreadTask implements Callable<Object> {

    private final Object invokeInstance;
    private final Object[] params;
    private final Method method;

    ThreadTask(Object invokeInstance, Object[] params, String methodName) throws NoSuchMethodException {
        this.invokeInstance = invokeInstance;
        this.params = params;
        method = getMethod(invokeInstance, methodName, params);
    }

    @Override
    public Object call() throws Exception {
        return this.method.invoke(invokeInstance, params);
    }

    private Method getMethod(Object invokerInstance, String methodName, Object... params) throws NoSuchMethodException, SecurityException {
        if (invokerInstance != null && methodName != null) {
            Method[] methods = invokerInstance.getClass().getMethods();
            for(int i = 0; i < methods.length; ++i) {
                Method method = methods[i];
                if (methodName.equals(method.getName())) {
                    Class<?>[] paramTypes = method.getParameterTypes();
                    if (params == null && paramTypes == null) {
                        return method;
                    }

                    if (params != null && paramTypes != null && paramTypes.length == params.length) {
                        for(int j = 0; j < params.length; ++j) {
                            if (params[j] != null && !paramTypes[j].isAssignableFrom(params[j].getClass())) {
                                throw new NoSuchMethodException();
                            }
                        }
                        return method;
                    }
                }
            }
            throw new NoSuchMethodException();
        } else {
            throw new NoSuchMethodException();
        }
    }
}
