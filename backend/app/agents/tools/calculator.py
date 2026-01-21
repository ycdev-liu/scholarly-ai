"""计算器工具。"""
import math
import re
import numexpr
from langchain_core.tools import BaseTool, tool


def calculator_func(expression: str) -> str:
    """使用numexpr计算数学表达式。

    当需要使用numexpr回答数学问题时很有用。
    此工具仅用于数学问题，不用于其他用途。仅输入
    数学表达式。

    Args:
        expression (str): 有效的numexpr格式的数学表达式。

    Returns:
        str: 数学表达式的结果。
    """
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # 限制对全局变量的访问
                local_dict=local_dict,  # 添加常见的数学函数
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"
