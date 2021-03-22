from typing import List, Dict, Tuple
import os
import numpy as np
from .exceptions import *
from .generation import GenerateManager, IdentityProducer
import pickle


class Sentence:
    def __init__(self, text: str):
        self.text = text

    def key(self):
        return self.text.strip().lower()


class Question(Sentence):
    def __init__(self, text: str, label: int, orignalParaPhrases: List[str] = None):
        super().__init__(text)
        self.label = label
        self.orignalParaPhrases = orignalParaPhrases or []

    def __str__(self) -> str:
        return "{} {}".format(self.text, self.label)


class Answer(Sentence):
    def __init__(self, text: str, label: int):
        super().__init__(text)
        self.label = label

    def __str__(self) -> str:
        return "{} {}".format(self.text, self.label)


class FAQUnit:
    def __init__(self, id: int, question: Question, orignal: Question, answer: Answer):
        """
        ID must be unique for each question(generated or orignal) !!!!
        question is the question itself , if the question is generated from some orignal 
        question then orignal will have the question , otherwise orignal will be same as the question
        answer ofcourse id the answer to the question 
        """
        assert question.label == orignal.label, "label for orignal and current question must be same"
        self.id = id
        self.question = question
        self.orignal = orignal
        self.answer = answer
        self.vectorRep = None  # To be assigned later using a model
        self.label = question.label

    def hasAssignedVector(self):
        if self.vectorRep is None:
            return False
        return True

    def __str__(self) -> str:
        return "Question ---> {}\nAnswer --->{}\n".format(self.question, self.answer)

    def __repr__(self) -> str:
        return self.__str__()


def processRaw(questions: List[str], answers: List[str]) -> Tuple[List[Question], List[Answer]]:
    assert len(questions) == len(answers)
    assert len(questions) == len(set(questions)), "Duplicate questions not allowed"

    # Now two or more questions may have the same answer !!! ,How ever it is not recommended !!!
    # You can add your own augmentations later !!!

    questions = [q.strip() for q in questions]
    answers = [a.strip() for a in answers]

    a2L = dict()
    label = 0
    outAnswers = []
    for answer in answers:
        answer_lowered = answer.lower()
        if answer_lowered not in a2L:
            a2L[answer_lowered] = label
            outAnswers.append(Answer(label=label, text=answer))
            label += 1

    l2Q: Dict[int, Question] = dict()
    for question, answer in zip(questions, answers):
        # Apply the text transformations here 
        label = a2L[answer.lower()]

        if label in l2Q:
            # HAndeling if multiple question map to the same answer !!!
            l2Q[label].orignalParaPhrases.append(question)
        else:
            # Creating new question
            l2Q[label] = Question(label=label, text=question)
        # print(label)
        # print(l2Q[label].orignalParaPhrases)

    outQuestions = list(l2Q.values())

    # for q in outQuestions:
    #    print(len(q.orignalParaPhrases))
    return outQuestions, outAnswers


class FAQ:
    def __init__(self, name: str, questions: List[str] = None, answers: List[str] = None):
        self.name = name

        self.questions: List[Question] = None
        self.answers: List[Answer] = None

        self.l2Q: Dict[int, Question] = dict()
        self.l2A: Dict[int, Answer] = dict()
        self.FAQ: List[FAQUnit] = None

        if not (questions is None or answers is None):
            assert len(questions) == len(answers)
            self.questions, self.answers = processRaw(questions=questions, answers=answers)
            self._runChecks()
            self._buildDicts()

        #########################################################################

    def _runChecks(self):
        """
        Running checks on questions and answers

        might add other checks like min/max sentence length etc
        """
        includedSet = set()
        for question in self.questions:
            if question.label in includedSet:
                raise WrongInputDataException(
                    "(ONLY FOR FAQ CLASS ERROR, this should be resolved in the processRaw method)Label for each question must be unique, found atleast 2 questions with same label")
            includedSet.add(question.label)

        answerlabelsSet = set()
        for answer in self.answers:
            if answer.label not in includedSet:
                raise WrongInputDataException(
                    "Label {} for answer {} does not match any question label , please remove this answer from all answer as it will never be answered anyway !!!".format(
                        answer.label, answer.text))

            if answer.label in answerlabelsSet:
                raise WrongInputDataException(
                    "Label for each answer must be unique (the text might be the same !!) but found at least two answers with label {}".format(
                        answer.label))

            answerlabelsSet.add(answer.label)

        # checking if any label in question points to no answer at all !!!
        differenceSet = includedSet - answerlabelsSet

        if len(differenceSet) > 0:
            raise WrongInputDataException(
                "The following questions labels have no corrosponding answer labels {} , please check the input data".format(
                    differenceSet))

    def _buildDicts(self):
        """
        To build label to question/answers mappings
        """
        if self.isEmpty():
            raise AttemptedUsingEmptyFAQ()

        for question in self.questions:
            self.l2Q[question.label] = question

        for answer in self.answers:
            self.l2A[answer.label] = answer

    def getAnswerWithLabel(self, label: int) -> Answer:
        """
        Returns answer for the particular label
        """
        if self.isEmpty():
            raise AttemptedUsingEmptyFAQ()

        if label not in self.l2A:
            raise LabelDoesNotExists("The answer with label {} does not exists".format(label))

        return self.l2A[label]

    def getQuestionWithLabel(self, label: int) -> Question:
        """
        Returns orignal question for the particular label
        """
        if self.isEmpty():
            raise AttemptedUsingEmptyFAQ()
        if label not in self.l2Q:
            raise LabelDoesNotExists("The question with label {} does not exists".format(label))

        return self.l2Q[label]

    def _paraphrase(self, generator: GenerateManager):
        """
        Takes the questions and answers and generated more question using the generator
        given , the number of questions to generate and other settings are to be applied in the 
        generator itself 

        Populated the self.FAQ property 
        """

        if self.isEmpty():
            raise AttemptedUsingEmptyFAQ()

        questionTexts = list(map(lambda q: q.text, self.questions))
        generatedQuestions = generator.generate(questionTexts)

        if len(generatedQuestions) != len(self.questions):
            raise WrongGeneratorOutputException(
                "number of orignal questions is {} but generator returned {} Lists !!!".format(len(self.questions),
                                                                                               len(generatedQuestions)))

        tempFAQ = []
        idCount = 0
        for orignalQuestion, paraphrases in zip(self.questions, generatedQuestions):
            label = orignalQuestion.label
            answer = self.getAnswerWithLabel(label)

            tempFAQ.append(FAQUnit(idCount, orignalQuestion, orignalQuestion, answer=answer))
            idCount += 1

            for parphrase in paraphrases:
                question = Question(label=label, text=parphrase)
                tempFAQ.append(FAQUnit(id=idCount, question=question, orignal=orignalQuestion, answer=answer))
                idCount += 1

            for orignalParaphrase in orignalQuestion.orignalParaPhrases:
                question = Question(label=label, text=orignalParaphrase)
                tempFAQ.append(FAQUnit(id=idCount, question=question, orignal=orignalQuestion, answer=answer))
                idCount += 1

        self.FAQ = tempFAQ

    def _assignVectors(self, model):
        """
        Using the model to assign vectors to each generated question
        """
        if self.isEmpty():
            raise AttemptedUsingEmptyFAQ()
        if not self.isUsable():
            raise AttemptedUsingNonUsableFAQ("Must generate paraphrases before assigning vectors")

        questions = [q.question.text for q in self.FAQ]
        vectors: List[np.ndarray] = model.encode(questions)

        if len(vectors) != len(questions):
            raise WrongModelOutputException(
                "The size of list of sentences input to model and that of output do not match , {} != {}".format(
                    len(questions, len(vectors))))

        for i, vector in enumerate(vectors):
            self.FAQ[i].vectorRep = vector

    def buildFAQ(self, generator: GenerateManager, model=None):
        """
        Will generate questions , and then the vectorrep of each question, 
        WILL NOT SAVE , MUST CALL SAVE 
        """
        if generator is None:
            generator = GenerateManager(producers=[IdentityProducer()], names=["IdentityProducer"], nums=[1])

        if self.isEmpty():
            raise AttemptedUsingEmptyFAQ()

        self._paraphrase(generator=generator)
        if model is not None:
            self._assignVectors(model=model)

    def isEmpty(self):
        if self.questions is None or self.answers is None or len(self.questions) == 0 or len(self.answers) == 0:
            return True

        return False

    def isUsable(self):
        if self.isEmpty():
            return False

        if self.FAQ is None or len(self.FAQ) == 0:
            return False
        return True

    def hasVectorsAssigned(self) -> bool:
        if not self.isUsable():
            return False
        for unit in self.FAQ:
            if not unit.hasAssignedVector():
                return False
        return True

    def load(self, rootDirPath):
        path = os.path.join(rootDirPath, self.name + ".pkl")
        with open(path, 'rb') as f:
            newObj = pickle.load(f)
        self.__dict__.update(newObj.__dict__)

    def save(self, rootDirPath):
        path = os.path.join(rootDirPath, self.name + ".pkl")
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def resetAssignedVectors(self):
        if self.FAQ is None:
            return

        for unit in self.FAQ:
            unit.vectorRep = None

    def resetFAQ(self):
        self.FAQ = None
        self.questions = None
        self.answers = None
        self.l2A = dict()
        self.l2Q = dict()

    def __len__(self):
        if self.isEmpty():
            raise AttemptedUsingEmptyFAQ()
        if not self.isUsable():
            raise ValueError(
                "cannot access length of a faq whose questions have not been generated !!! MUST CALL BUILD FAQ BEFORE THIS, OR LOAD FAQ FROM PREEXISTING SOURCE")

        return len(self.FAQ)


class FAQOutput:
    def __init__(self, answer: Answer, question: Question, faqName: str, faqId: int, score: float,
                 similarQuestions: List[str], maxScore: float):
        self.answer = answer
        self.question = question
        self.faqId = faqId
        self.faqName = faqName
        self.score = score
        self.similarQuestions = similarQuestions
        self.maxScore = maxScore

    def __str__(self):
        out = self
        return "faqName - {}\n\nanswer - {}\n\nquestion - {}\n\nmaxScore - {}\n\nscore - {}\n\n{}\n\n{}".format(
            out.faqName, out.answer.text, out.question.text, out.maxScore, out.score, "=" * 50, out.similarQuestions)
