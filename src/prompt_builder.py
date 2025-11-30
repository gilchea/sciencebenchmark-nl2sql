from typing import List, Dict

BASE_INSTRUCTION = """You are an NL-to-SQL assistant. Given a natural language question and the database schema, generate a single SQL query that answers the question. Conditions:
- Use SQL compatible with PostgreSQL 9.5.
- Do not add comments.
- Keep it concise: one SQL statement ending with a semicolon.
- Use the schema provided."""

EXAMPLE_FMT = """### Example
Question: {question}
SQL: {sql}
"""

PROMPT_TEMPLATE = """{schema_context}

{instruction}

{icl_block}
Question: {question}
SQL:
"""

class PromptBuilder:
    """
    A class to build structured prompts for the NL-to-SQL task.
    """
    def __init__(self,
                 base_instruction: str = BASE_INSTRUCTION,
                 example_fmt: str = EXAMPLE_FMT,
                 prompt_template: str = PROMPT_TEMPLATE):
        """
        Initializes the PromptBuilder with prompt templates.
        """
        self.base_instruction = base_instruction
        self.example_fmt = example_fmt
        self.prompt_template = prompt_template

    def build(self,
              schema_context: str,
              question: str,
              icl_examples: List[Dict[str, str]]) -> str:
        """
        Builds the final prompt string.
        """

        # 1. Xây dựng khối ICL
        if icl_examples:
            icl_block_list = []
            for ex in icl_examples:
                sql_string = ex.get('query')
                if not sql_string:
                    sql_string = ex.get('sql', '-- ERROR: No SQL query found in example')

                icl_block_list.append(
                   self.example_fmt.format(question=ex['question'], sql=sql_string)
                )

            icl_block = "\n".join(icl_block_list)
        else:
            icl_block = ""

        # 2. Định dạng prompt cuối cùng
        final_prompt = self.prompt_template.format(
            schema_context=f"### Database Schema\n{schema_context}\n",
            instruction=self.base_instruction,
            icl_block=icl_block.strip() + "\n" if icl_block else "",
            question=question
        ).strip()

        return final_prompt
