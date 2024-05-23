PROMPT_W_CONTEXT = 'You are a helpful assistant, below is a query from a user and some relevant context to help you answer. \
Answer the user\'s question. Your answer should as short and concise as possible. \
If you cannot find the answer to the question, just say "I don\'t know".\n \
\n\n\nE.g.:\n\nQuery: How many days are there in a year? \n\nAnswer:365. \
\n\n\nE.g.:\n\nQuery: Who wrote "The Catcher in the Rye"? \n\nAnswer:J.D. Salinger. \
\n\n\n\nContext:\n[context] \n\nQuery: [question] \n\nAnswer:'

PROMPT_WO_CONTEXT = 'You are a helpful assistant, below is a query from a user. \
Answer the user\'s question. Your answer should as short and concise as possible. \
If you cannot find the answer to the question, just say "I don\'t know".\n \
\n\n\nE.g.:\n\nQuery: How many days are there in a year? \n\nAnswer:365. \
\n\n\nE.g.:\n\nQuery: Who wrote "The Catcher in the Rye"? \n\nAnswer:J.D. Salinger. \
\n\n\n\nQuery: [question] \n\nAnswer:'


def wrap_prompt(question, context) -> str:
    if context is None:
        return PROMPT_WO_CONTEXT.replace("[question]", question)
    if isinstance(context, list):
        context = "\n".join(context)

    return PROMPT_W_CONTEXT.replace("[question]", question).replace(
        "[context]", context
    )
