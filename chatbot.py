from responses import responses

EXIT_COMMANDS = ["bye", "exit", "quit"]

def process(user_input):
    reply = responses.get(user_input,"I do not understand.")
    return reply



print("---RULE-BASED AI CHATBOT---")
print("Type [exit,bye,quit] to exit")



while True:

    raw_input_text = input("You: ")
    clean_input = raw_input_text.lower().strip()
    if clean_input in EXIT_COMMANDS:
        print("GoodBye!")
        break
    response = process(clean_input)
    print("Bot:", response)