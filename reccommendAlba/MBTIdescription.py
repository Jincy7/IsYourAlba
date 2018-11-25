class MBTI_type():
    def __init__(self, type):
        self.type = type

    def __str__(self):
        if self.type == "ISTJ":
            return "신중하고, 조용하며 집중력이 강하고 매사에 철저하다. 위기상황에서도 침착하고, 충동적으로 일을 처리하지 않는다."
        elif self.type == "ISFJ":
            return "성실하고, 온화하며, 협조를 잘한다. 침착성과 인내심은 가정이나 집단에 안정감을 준다."
        elif self.type == "INFJ":
            return "창의력, 통찰력이 뛰어나며, 직관력으로 말없이 타인에게 영향력 대인관계를 형성할 때는 진실한 관계를 맺고자 한다."
        elif self.type == "INTJ":
            return "행동과 사고에 있어서 독창적이다. 아이디어와 목표를 달성하는데 강한 추진력을 가지고 있다."
        elif self.type == "ISTP":
            return "말이 없으며, 논리적이고 객관적으로 인생을 관찰하는 형. 적응력이 강하고 새롭고 긴급한 일을 잘 다룬다."
        elif self.type == "ISFP":
            return "말없이 다정하고 친절하며, 겸손하다. 자연, 사물, 예술 등을 감상하는 능력과 식별력이 뛰어 나며,자연과 동물을 사랑한다."
        elif self.type == "INFP":
            return "마음이 따뜻하고 조용하며 자신의 일에 책임감이 강하고 성실. 열성을 가진 분야는 설득력 있고 독창적일 수 있다."
        elif self.type == "INTP":
            return "과묵하나 관심이 있는 분야에 대해서는 말을 잘한다. 이해가 빠르고 높은 직관력과 통찰력 및 지적 호기심이 많다."
        elif self.type == "ESTP":
            return "친구, 운동, 음식 등  다양한 활동을 좋아 한다. 예술적인 멋과 판단력을 지니며, 타고난 재치와 사교력이 있다."
        elif self.type == "ESFP":
            return "현실적이고 실제적이며 친절하다. 어떤 상황이든 잘 적응하며 수용력이 강하고 사교적이다."
        elif self.type == "ENFP":
            return " 열성적이고, 창의적이며, 풍부한 상상력과 새로운 일을 잘 시작함. 관심 있는 일이면 무엇이든지 척척 해내는 열성파이다."
        elif self.type == "ENTP":
            return "독창적이며 창의력이 풍부하고 넓은 안목을 갖고 있고 다방면에 재능이 많다. 자신감이 많다."
        elif self.type == "ESTJ":
            return "일을 만들고 계획하고, 추진하는데 뛰어난 능력을 가지고 있다. 친구나 주변사람을 배려하는 리더역할을 잘한다."
        elif self.type == "ESFJ":
            return "동정심이 많고 다른 사람에게 관심을 쏟고 협동을 중시 한다. 양심적이고, 정리정돈을 잘하며, 참을성이 많다."
        elif self.type == "ENFJ":
            return "동정심이 많고, 인화를 중시하며, 민첩하고 성실하다. 참을성이 많다. 사교성이 풍부하고 인기 있다."
        elif self.type == "ENTJ":
            return "열성이 많고 솔직하고 단호하고 통솔력이 있다. 정보에 밝고, 지식에 대한 관심과 욕구가 많다."
        else:
            return "성격이 정해지지 않았습니다! 다시 시도해보세요."


def putItme():
    types = {}
    types["ISTJ"] = MBTI_type("ISTJ")
    types["ISFJ"] = MBTI_type("ISFJ")
    types["INFJ"] = MBTI_type("INFJ")
    types["INTJ"] = MBTI_type("INTJ")
    types["ISTP"] = MBTI_type("ISTP")
    types["ISFP"] = MBTI_type("ISFP")
    types["INFP"] = MBTI_type("INFP")
    types["INTP"] = MBTI_type("INTP")
    types["ESTP"] = MBTI_type("ESTP")
    types["ESFP"] = MBTI_type("ESFP")
    types["ENFP"] = MBTI_type("ENFP")
    types["ESTJ"] = MBTI_type("ESTJ")
    types["ESFJ"] = MBTI_type("ESFJ")
    types["ENFJ"] = MBTI_type("ENFJ")
    types["ENTJ"] = MBTI_type("ENTJ")
    types["ENTP"] = MBTI_type("ENTP")
    return types


types = putItme()
