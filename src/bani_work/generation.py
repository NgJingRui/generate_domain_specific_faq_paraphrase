from typing import Any, List, Dict, Tuple
import warnings


class QuestionGenerator:
    def __init__(self, name: str, producer, num: int):
        self.name = name

        if "exact_batch_generate" not in dir(producer) and "batch_generate" not in dir(producer):
            raise AttributeError("Must have either batch_generate or exact batch generate in the producer object !!")
        self.producer = producer
        self.num = num

    def generate(self, questions: List[str]) -> Dict[str, List[str]]:
        """
        takes as input a list of questions

        the producer is an object that has a method , batch_generate

        batch generate takes in a list of questions as input and 
        outputs a dict .....
        the dict maps each orignal question to a list of generated questions
        """
        print("working with {} pipeline".format(self.name))
        if "exact_batch_generate" in dir(self.producer):
            result_dict = self.producer.exact_batch_generate(questions, self.num)
        else:
            result_dict = self.producer.batch_generate(questions)
        if result_dict is None:
            answer = dict()
            for q in questions:
                answer[q] = []
            return answer

        for orignal_question in result_dict:
            result_dict[orignal_question] = result_dict[orignal_question][:self.num]

        return result_dict


class IdentityProducer:
    """ 
    Use this generator when you do not need to use the generation pipeline !!!
    This will generate nothing and just return empty array !!!
    """

    def __init__(self):
        pass

    def batch_generate(self, questions: List[str]) -> Dict[str, List[str]]:
        resultDict = dict()
        for question in questions:
            resultDict[question] = []

        return resultDict


class GenerateManager:
    """
    Implement here functions that will help multiprocessing , moniter timing etc
    """

    def __init__(self, producers: List[Any], names: List[str] = None, nums: List[int] = None):
        """
        each producer is a user defined object that either implements 
        batch_generate or exact_batch_generate 
        """
        if names is None:
            names = []
            for i in range(len(producers)):
                names.append("generator" + str(i))

        assert len(names) == len(set(names)), "The names of generators must be unique !!!"

        if nums is None:
            nums = [10 for i in range(len(producers))]

        assert len(producers) == len(names) == len(nums)
        self.generators: List[QuestionGenerator] = [QuestionGenerator(name=names[i], producer=producers[i], num=nums[i])
                                                    for i in range(len(producers))]

    def removeProducer(self, name):
        preNames = [generator.name for generator in self.generators]
        if name not in preNames:
            raise ValueError("A producer with name {} does not exists exists".format(name))
        newGenerators: List[QuestionGenerator] = []
        for generator in self.generators:
            if generator.name != name:
                newGenerators.append(generator)
        self.generators = newGenerators

    def addProducer(self, producer, name: str, toGenerate: int):
        preNames = [generator.name for generator in self.generators]
        if name in preNames:
            raise ValueError("A producer with name {} already exists".format(name))
        newGenerator = QuestionGenerator(name=name, producer=producer, num=toGenerate)
        self.generators.append(newGenerator)

    def producerList(self) -> Tuple[List[str], List[int], List[Any]]:
        names = []
        nums = []
        producers = []
        for generator in self.generators:
            names.append(generator.name)
            nums.append(generator.num)
            producers.append(generator.producer)

        return names, nums, producers

    def generate(self, questions: List[str]) -> List[List[str]]:
        """
        takes the questions given , and uses the generators to generate.
        Returns generated questions in order !!!
        """

        mergedDict = self._sequential(questions)

        result = []

        for question in questions:
            if question in mergedDict:
                result.append(mergedDict[question])
            else:
                warnings.warn("Zero questions generated for ---> {}".format(question))
                result.append([])

        return result

    def _mergeDicts(self, dicts: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        mergedDict = dict()
        for dct in dicts:
            for que, gen_ques in dct.items():
                if que not in mergedDict:
                    mergedDict[que] = []
                mergedDict[que].extend(gen_ques)

        return mergedDict

    def _sequential(self, questions):
        """
        Applies generators in a series , ie one after another no 
        multiprocessing or multithreading 
        """

        resultDicts = []

        for generator in self.generators:
            try:
                resultDicts.append(generator.generate(questions))
            except Exception as exception:
                warnings.warn("{} pipeline failed !!!! {}".format(generator.name, exception))

        return self._mergeDicts(resultDicts)
