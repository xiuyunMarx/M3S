import os
import glob
import urllib.parse

# ================= 配置区域 =================
EXP_FOLDERS = [
    "CFG10.0_sfg15.0_skip10",
    "CFG15.0_sfg10.0_skip10",  
    "CFG15.0_sfg15.0_skip10", 
    "CFG15.0_sfg20.0_skip10",
    "CFG17.5_sfg15.0_skip10",
    "CFG17.5_sfg17.5_skip10",
    "CFG17.5_sfg20.0_skip10",
    "CFG20.0_sfg10.0_skip10",
    "CFG20.0_sfg15.0_skip10", 
    "CFG20.0_sfg20.0_skip10",
]

TARGET_STYLE = "style1_0_int_0.1"

# 选取几个代表性的 Prompt
SELECTED_PROMPTS = [
    # 1. 动物与细节 (测试毛发和微小纹理)
    "a cat sitting inside a teacup",
    "a fox standing on a log",
    "a squirrel holding an acorn",
    "two white bunnies",
    "a dragon flying in the sky, full body", # 复杂生物结构

    # 2. 人物与动作 (测试肢体结构和人脸是否崩坏)
    "James Bond in a tuxedo and holding a gun",
    "a man performing tai chi",
    "a person doing yoga pose",
    "a portrait of naruto", # 测试特定角色的还原度

    # 3. 风景与大结构 (测试画面构图和风格氛围)
    "a mountain peak piercing clouds",
    "a stone arch bridge at dawn",
    "a waterfall cascading down rocks",
    "enchanted forest with glowing mushrooms",
    "sydney opera house", # 测试几何建筑是否被风格扭曲

    # 4. 简单物体 (测试是否有不必要的噪点或过度纹理化)
    "a key lying on a wooden table",
    "a cactus in a clay pot",
    "a steaming coffee cup on a saucer",
    "a vintage camera on a tripod",
    "a single rose in a slim vase",
    
    # 5. 抽象/特殊视角
    "a winding staircase in a tower"
]

IMG_FILENAME = "out_transfer---seed_42.png"
# ===========================================

def generate_html():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: sans-serif; background: #f0f0f0; margin: 20px; }
            h2 { text-align: center; }
            table { border-collapse: collapse; width: 100%; table-layout: fixed; }
            th, td { border: 1px solid #999; padding: 5px; text-align: center; vertical-align: middle; background: white;}
            th { background-color: #333; color: white; position: sticky; top: 0; z-index: 10; padding: 10px;}
            img { width: 100%; height: auto; display: block; }
            .prompt-col { width: 120px; font-weight: bold; font-size: 14px; word-wrap: break-word; background: #e8e8e8; }
            .not-found { color: red; font-weight: bold; }
            .debug-info { font-size: 10px; color: #aaa; margin-top: 5px; word-break: break-all;}
        </style>
    </head>
    <body>
        <h2>Parameter Comparison (Style: """ + TARGET_STYLE + """)</h2>
        <table>
            <thead>
                <tr>
                    <th class="prompt-col">Prompt</th>
    """
    
    for folder in EXP_FOLDERS:
        short_name = folder.replace("_skip10", "").replace(".0", "")
        html_content += f"<th>{short_name}</th>"
    
    html_content += "</tr></thead><tbody>"

    for prompt in SELECTED_PROMPTS:
        html_content += f"<tr><td class='prompt-col'>{prompt}</td>"
        
        for folder in EXP_FOLDERS:
            # 1. 构建 Prompt 所在的目录路径
            prompt_dir = os.path.join(folder, TARGET_STYLE, prompt)
            
            # 2. 寻找中间那层 "app=..." 的文件夹
            # 使用 glob 查找 prompt_dir 下的所有子文件夹
            found_img_rel_path = None
            
            if os.path.exists(prompt_dir):
                # 查找该目录下所有的子项
                subitems = os.listdir(prompt_dir)
                for item in subitems:
                    # 检查是不是目录（比如 app=13---struct=14）
                    full_sub_path = os.path.join(prompt_dir, item)
                    if os.path.isdir(full_sub_path):
                        # 检查图片是否在这个子目录里
                        target_img = os.path.join(full_sub_path, IMG_FILENAME)
                        if os.path.exists(target_img):
                            # 找到了！记录相对路径用于 HTML
                            # 注意：HTML URL 需要对路径中的特殊字符（如空格）进行转义
                            # 但在这里我们手动拼接路径字符串
                            found_img_rel_path = os.path.join(prompt, item, IMG_FILENAME)
                            found_img_rel_path = os.path.join(folder, TARGET_STYLE, found_img_rel_path)
                            break
            
            if found_img_rel_path:
                # 对路径中的空格等进行 URL 编码
                # 也就是把 "a cat" 变成 "a%20cat"
                url_path = urllib.parse.quote(found_img_rel_path)
                
                html_content += f"""
                    <td>
                        <a href="{url_path}" target="_blank">
                            <img src="{url_path}" loading="lazy" alt="Image">
                        </a>
                    </td>
                """
            else:
                html_content += f"<td class='not-found'>Not Found<div class='debug-info'>Checked: {prompt_dir}/*/{IMG_FILENAME}</div></td>"
        
        html_content += "</tr>"

    html_content += "</tbody></table></body></html>"

    with open("comparison.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("生成完成！修复了子目录路径问题。")

if __name__ == "__main__":
    generate_html()