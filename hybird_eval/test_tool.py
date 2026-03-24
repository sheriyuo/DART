import json
import asyncio
from hybird_eval.hybird_inference import ToolServerClient


async def test_tool():
    # 注意：这里的端口需要和你启动脚本里随机生成的端口一致
    # 你可以在启动脚本里把 `echo $port` 输出出来，或者手动指定一个固定端口测试
    SERVER_PORT = 30270  
    SERVER_URL = f"http://localhost:{SERVER_PORT}"

    # 初始化客户端
    tool_client = ToolServerClient(SERVER_URL)
    
    try:
        # 1. 先检查服务是否健康
        print("Checking health...")
        health = await tool_client.check_health()
        print(f"Health status: {json.dumps(health, indent=2)}")

        if health.get("status") != "healthy":
            print("Server is not ready.")
            return

        # 2. 模拟调用 search_retrieval 工具
        # 假设 actions 就是搜索的 query
        queries = [
            "<search> What implies that the function is explicitly defined? </search>",
            "<search> Who matches the description of the 'Father of Artificial Intelligence'? </search>"
        ]
        
        print(f"\nSending {len(queries)} queries...")
        
        result = await tool_client.get_observations(actions=queries)
        
        # 3. 解析结果
        observations = result.get("observations", [])
        dones = result.get("dones", [])
        valids = result.get("valids", [])
        
        for i, (obs, valid) in enumerate(zip(observations, valids)):
            print(f"\n--- Result {i+1} (Valid: {valid}) ---")
            print(f"Query: {queries[i]}")
            # 检索工具返回的通常是一个列表或字典字符串，这里打印部分内容
            print(f"Observation: {str(obs)[:200]}...") 

    finally:
        # 记得关闭连接
        await tool_client.close()

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(test_tool())