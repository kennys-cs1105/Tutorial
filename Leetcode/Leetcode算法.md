# LeetCode刷题

*Created by KennyS*

LeetCode算法

---


## 记录

### 1. 数组

#### 1.1 数组理论基础

1.1.1 数组是存放在连续内存空间上的相同类型数据的集合

- 数组下标都是从0开始
- 数组内存空间的地址是连续的

1.1.2 数组内存空间是连续的, 所以在删除和增添元素的时候, 需要移动其他元素的地址

注意：C++中, 需要注意vector和array的区别, vector的底层实现是array, 严格讲vector是容器, 不是数组

1.1.3 数组元素不能删除, 只能覆盖

参考二维数组

1.1.4 二维数组在C++中的空间地址

```cpp
void test_arr(){
    int array[2][3] = {
        {0, 1, 2},
        {3, 4, 5}
    };
    cout << &array[0][0] << " " << &array[0][1] << " " << &array[0][2] << endl;
    cout << &array[1][0] << " " << &array[1][1] << " " << &array[1][2] << endl;
}

int main(){
    test_arr();
}
```

输出:
> 0x7ffee4065820 0x7ffee4065824 0x7ffee4065828
0x7ffee406582c 0x7ffee4065830 0x7ffee4065834

注意：地址为16进制, 相邻元素就差了4个字节, 由于这是个int型数组, 故连续

---

### 1.2 二分查找

**问题**：给定一个n个元素的有序（升序）整型数组nums, 和一个目标值target, 写一个函数搜索nums中的target, 如果目标值存在, 则返回下标, 否则返回-1

```
示例1：
输入： nums = [-1, 0, 3, 5, 9, 12], target = 9
输出： 4
```

```
示例2：
输入： nums = [-1, 0, 3, 5, 9, 12], target = 2
输出： -1
```

**提示**：
- 假设nums中的元素是不重复的
- n在[1, 10000]之间
- nums的每个元素都在[-9999, 9999]之间

**解题思路：**

1. **前提是数组为有序数组, 数组中无重复元素**, 一旦有重复元素, 二分法返回的下标可能不唯一, 这些都是二分法使用的前提条件

2. 二分查找涉及很多的边界条件, 需要考虑 `while(left < right)`, `while(left <= right)`, `right = middle`, `right = middle - 1`

3. 区间的定义就是不变量, 在二分查找的过程中保持不变量, 在while寻找中每一次边界的处理都要坚持根据区间的定义来操作

4. 二分法区间定义分为两种：左闭右闭 -> [left, right]; 左闭右开 -> [left, right)


**两种写法**


1. target定义在左闭右闭 -> [left, right]

因为target在[left, right]区间中, 所以有如下两点：

- while(left <= right), 因为left = right是有意义的
- if(nums[middle] > target), right要赋值为middle-1, 因为当前这个nums[middle]一定不是target, 那么接下来要查找的左区间结束下标位置就是middle-1 

```cpp
class Solution{
public:
    int search(vector<int>& nums, int target){
        int left = 0;
        int right = nums.size() - 1; // [left, right]
        while(left <= right){
            int middle = (left + right) / 2;
            if (nums[middle] > target){
                right = middle - 1;
            } else if (nums[middle] < target){
                left = middle + 1;
            } else {
                return middle;
            }
        }
        return -1;
    }
};
```

```python
class Solution:
    def search(self, nums:List[int], target:int) -> int:
        left, right = 0, len(nums) - 1 
        while(left <= right):
            middle = (left + right) // 2
            if (nums[middle] > target):
                right = middle - 1
            elif (nums[middle] < target):
                left = middle + 1;
            else:
                return middle;
        return -1
```

2. target定义在左闭右开 -> [left, right)

有如下两点

- while(left < right), left无法等于right
- if (nums[middle] > target), right更新为middle, 因为当前nums[middle]不等于target, 去左区间继续寻找, 而寻找区间是左闭右开区间, 所以right更新为middle, 即：下一个查询区间不会去比较

```cpp
class Solution{
public:
    int search(vector<int>& nums, int target){
        int left = 0;
        int right = nums.size(); // [left, right)
        while(left < right){
            int middle = (left + right) / 2;
            if (nums[middle] > target){
                right = middle;
            } else if (nums[middle] < target){
                left = middle + 1;
            } else {
                return middle;
            }
        }
        return -1;
    }
};
```



### 1.3 移除元素

**问题**：给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度

**解题思路**：

1. 数组是连续的、元素类型相同的数据结构

2. 移除某一个元素，需要将其覆盖，而不是直接删除，对于最后个元素不做处理

3. 类似于`vector.size`中的`erase`函数，是个O(n)操作


**暴力解法**

```cpp
for (数组大小){
    for (target后的元素, 向前覆盖)
}
```

**双指针思路**

- 新数组中不包含`target`
- 同一数组构建两个指针
    - fast: 新数组需要的元素
    - slow: 新数组下标

```cpp
int slow = 0;
for (int fast = 0; fast < nums.size(); fast++) {
    if (nums[fast] != target){
        nums[slow] = nums[fast];
        slow++;
    }
}
return slow;
```