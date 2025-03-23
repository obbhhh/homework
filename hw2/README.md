**测试当前服务器上单核浮点预算峰值**
当前cpu信息如下：
![cpu_info](./images/cpu_info.png)

当前cpu为 x86-64 架构，运行 ./cpufp/build_x64.sh 编译 benchmark 后 ./cpufp/cpufp --thread_pool=[0] 测试单核浮点预算峰值结果如下：
![singe cpu peak](./images/singel%20cpu%20peak.png)

