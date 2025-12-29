#!/bin/bash

# 设置要搜索的目标文件夹名称
TARGET_DIR="latents"

echo "正在搜索当前目录及子目录下所有的 '$TARGET_DIR' 文件夹..."

# 1. 查找并统计
# find . -type d -name "latents"
# -type d: 只找文件夹
# -name: 名字匹配
files_to_delete=$(find . -type d -name "$TARGET_DIR")
count=$(echo "$files_to_delete" | grep -v "^$" | wc -l)

if [ "$count" -eq 0 ]; then
    echo "太棒了，没有找到任何 '$TARGET_DIR' 文件夹，不需要清理。"
    exit 0
fi

# 2. 显示即将删除的列表（为了不刷屏，只显示前10个）
echo "------------------------------------------------"
echo "共发现 $count 个 '$TARGET_DIR' 文件夹。"
echo "示例路径 (前10个):"
echo "$files_to_delete" | head -n 10
if [ "$count" -gt 10 ]; then
    echo "... (以及其他 $(($count - 10)) 个)"
fi
echo "------------------------------------------------"
echo "警告: 这些文件夹及其内部的所有 .pt 文件将被永久删除！"

# 3. 交互式确认
read -p "你确定要删除它们吗? (y/n): " -n 1 -r
echo    # 换行

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "正在执行清理..."
    # -exec rm -rf {} + 是最快且最标准的删除方式
    find . -type d -name "$TARGET_DIR" -exec rm -rf {} +
    echo "清理完成！磁盘空间已释放。"
else
    echo "操作已取消。"
fi
