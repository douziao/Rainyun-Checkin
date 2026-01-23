"""
服务器管理模块
负责服务器到期检查、自动续费等业务逻辑
"""
import logging
import os
from datetime import datetime
from typing import Optional

from api_client import RainyunAPI, RainyunAPIError

logger = logging.getLogger(__name__)

# 续费成本：7天 = 2258 积分（固定值）
RENEW_COST_7_DAYS = 2258


class ServerInfo:
    """服务器信息"""

    def __init__(self, server_id: int, name: str, expired_at: int):
        self.id = server_id
        self.name = name
        self.expired_at = expired_at  # Unix 时间戳

    @property
    def expired_datetime(self) -> datetime:
        """到期时间（datetime 对象）"""
        return datetime.fromtimestamp(self.expired_at)

    @property
    def days_remaining(self) -> int:
        """剩余天数"""
        delta = self.expired_datetime - datetime.now()
        return max(0, delta.days)

    @property
    def expired_str(self) -> str:
        """到期时间格式化字符串"""
        return self.expired_datetime.strftime("%Y-%m-%d %H:%M:%S")


class ServerManager:
    """服务器管理器"""

    def __init__(self, api_key: str):
        """
        初始化服务器管理器

        Args:
            api_key: 雨云 API 密钥
        """
        self.api = RainyunAPI(api_key)
        # 从环境变量读取配置
        self.auto_renew = os.environ.get("AUTO_RENEW", "true").lower() == "true"
        # 修复：RENEW_THRESHOLD_DAYS 类型错误时给出明确提示
        try:
            self.renew_threshold = int(os.environ.get("RENEW_THRESHOLD_DAYS", "7"))
        except ValueError:
            logger.error("配置错误：RENEW_THRESHOLD_DAYS 必须是整数，使用默认值 7")
            self.renew_threshold = 7

        # 白名单模式：只续费指定的产品ID（逗号分隔，为空则续费所有）
        renew_ids_str = os.environ.get("RENEW_PRODUCT_IDS", "").strip()
        if renew_ids_str:
            try:
                self.renew_product_ids = [int(x.strip()) for x in renew_ids_str.split(",") if x.strip()]
                logger.info(f"白名单模式：只续费产品 {self.renew_product_ids}")
            except ValueError:
                logger.error("配置错误：RENEW_PRODUCT_IDS 格式无效，应为逗号分隔的数字")
                self.renew_product_ids = []
        else:
            self.renew_product_ids = []  # 空列表表示续费所有

    def get_all_servers(self) -> list:
        """
        获取所有服务器信息

        Returns:
            ServerInfo 对象列表
        """
        servers = []
        try:
            server_ids = self.api.get_server_ids()
            logger.info(f"找到 {len(server_ids)} 台服务器")

            for sid in server_ids:
                try:
                    detail = self.api.get_server_detail(sid)
                    # API 返回格式：{"Data": {"ExpDate": 1770306863, ...}}
                    server_data = detail.get("Data", {})
                    expired_at = server_data.get("ExpDate", 0)
                    # 修复：ExpDate 缺失或无效时跳过该服务器，避免误续费
                    if not expired_at or expired_at <= 0:
                        logger.warning(f"服务器 {sid} 的 ExpDate 无效 ({expired_at})，跳过")
                        continue
                    # 服务器名：尝试从 EggType 获取，否则用默认名
                    egg_info = server_data.get("EggType", {}).get("egg", {})
                    server_name = egg_info.get("title", f"游戏云-{sid}")
                    server = ServerInfo(
                        server_id=sid,
                        name=server_name,
                        expired_at=expired_at
                    )
                    servers.append(server)
                    logger.info(f"  - {server.name}: 到期 {server.expired_str}, 剩余 {server.days_remaining} 天")
                except RainyunAPIError as e:
                    logger.error(f"获取服务器 {sid} 详情失败: {e}")

        except RainyunAPIError as e:
            logger.error(f"获取服务器列表失败: {e}")

        return servers

    def check_and_renew(self) -> dict:
        """
        检查所有服务器到期时间，必要时自动续费

        Returns:
            结果摘要字典：
            {
                "points": 当前积分,
                "servers": [服务器状态列表],
                "renewed": [续费成功的服务器],
                "warnings": [警告信息]
            }
        """
        result = {
            "points": 0,
            "servers": [],
            "renewed": [],
            "warnings": []
        }

        try:
            # 获取当前积分
            result["points"] = self.api.get_user_points()
            logger.info(f"当前积分: {result['points']}")

            # 获取所有服务器
            servers = self.get_all_servers()

            for server in servers:
                server_status = {
                    "name": server.name,
                    "expired": server.expired_str,
                    "days_remaining": server.days_remaining,
                    "renewed": False
                }

                # 检查是否需要续费
                if server.days_remaining <= self.renew_threshold:
                    logger.warning(f"⚠️ {server.name} 即将到期！剩余 {server.days_remaining} 天")

                    # 白名单检查：如果设置了白名单，只续费白名单内的产品
                    if self.renew_product_ids and server.id not in self.renew_product_ids:
                        logger.info(f"  ↳ 跳过：不在白名单中 (ID: {server.id})")
                        result["warnings"].append(f"{server.name} 即将到期，但不在续费白名单中")
                    elif self.auto_renew:
                        # 检查积分是否足够
                        if result["points"] >= RENEW_COST_7_DAYS:
                            try:
                                self.api.renew_server(server.id, days=7)
                                logger.info(f"✅ {server.name} 续费成功！消耗 {RENEW_COST_7_DAYS} 积分")
                                result["points"] -= RENEW_COST_7_DAYS
                                server_status["renewed"] = True
                                result["renewed"].append(server.name)
                            except RainyunAPIError as e:
                                logger.error(f"❌ {server.name} 续费失败: {e}")
                                result["warnings"].append(f"{server.name} 续费失败: {e}")
                        else:
                            warning = f"积分不足！需要 {RENEW_COST_7_DAYS}，当前 {result['points']}"
                            logger.warning(warning)
                            result["warnings"].append(warning)
                    else:
                        result["warnings"].append(f"{server.name} 即将到期，但自动续费已关闭")

                result["servers"].append(server_status)

        except RainyunAPIError as e:
            logger.error(f"服务器检查失败: {e}")
            result["warnings"].append(f"API 调用失败: {e}")

        return result

    def generate_report(self, result: dict) -> str:
        """
        生成服务器状态报告（用于通知推送）

        Args:
            result: check_and_renew 返回的结果字典

        Returns:
            格式化的报告字符串
        """
        lines = [
            "━━━━━━ 服务器状态 ━━━━━━",
            f"💰 当前积分: {result['points']}"
        ]

        if result["servers"]:
            lines.append("")
            for s in result["servers"]:
                status = "✅ 已续费" if s["renewed"] else ""
                days_emoji = "🔴" if s["days_remaining"] <= 3 else "🟡" if s["days_remaining"] <= 7 else "🟢"
                lines.append(f"🖥️ {s['name']}")
                lines.append(f"   {days_emoji} 剩余 {s['days_remaining']} 天 ({s['expired']}) {status}")
        else:
            lines.append("📭 无服务器")

        if result["renewed"]:
            lines.append("")
            lines.append(f"🎉 本次续费: {', '.join(result['renewed'])}")

        if result["warnings"]:
            lines.append("")
            lines.append("⚠️ 警告:")
            for w in result["warnings"]:
                lines.append(f"   - {w}")

        return "\n".join(lines)
