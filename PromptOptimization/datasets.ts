export type MCExample = { // PiQA, HellaSwag
    question: string; 
    choices: string[]; 
    correct: string;  // "A", "B", "C", or "D"
  };
  
  export type BoolQExample = {
    question: string;
    passage: string;
    answer: boolean;
  };
  
  export type GSM8KExample = {
    question: string;
    answer: number;
  };
  
  // load the datasets from data/ folder
  export async function loadPIQA(maxN = 100): Promise<MCExample[]> {
    try {
      const txt = await Deno.readTextFile('data/piqa.json');
      const rows = JSON.parse(txt);
      
      return rows.slice(0, maxN).map((r: any) => ({
        question: r.goal,
        choices: [r.sol1, r.sol2],
        correct: r.label === '1' ? 'B' : 'A',
      }));
    } catch (error) {
      console.warn("Error using mock data");
      return generateDummyPIQA(Math.min(maxN, 10));
    }
  }
  
  function generateDummyPIQA(n: number): MCExample[] {
    return Array.from({ length: n }, (_, i) => ({
      question: `How do you accomplish task ${i}?`,
      choices: [`Method A for ${i}`, `Method B for ${i}`],
      correct: Math.random() > 0.5 ? 'A' : 'B'
    }));
  }
  
  export async function loadHellaSwag(maxN = 100): Promise<MCExample[]> {
    try {
      const txt = await Deno.readTextFile('data/hellaswag.json');
      const rows = JSON.parse(txt);
      
      return rows.slice(0, maxN).map((r: any) => {
        const labelMap: Record<string, string> = { '0': 'A', '1': 'B', '2': 'C', '3': 'D' };
        return {
          question: r.context,
          choices: r.endings,  // Array of 4 endings
          correct: labelMap[r.label] || 'A',
        };
      });
    } catch (error) {
        console.warn("Error using mock data");
        return generateDummyHellaSwag(Math.min(maxN, 10));
    }
  }
  
  function generateDummyHellaSwag(n: number): MCExample[] {
    return Array.from({ length: n }, (_, i) => ({
      question: `A person is doing activity ${i}. They then`,
      choices: [
        `continue with step A`,
        `move to step B`,
        `switch to step C`,
        `finish with step D`
      ],
      correct: ['A', 'B', 'C', 'D'][Math.floor(Math.random() * 4)]
    }));
  }
  
  export async function loadBoolQ(maxN = 100): Promise<BoolQExample[]> {
    try {
      const txt = await Deno.readTextFile('data/boolq.json');
      const rows = JSON.parse(txt);
      
      return rows.slice(0, maxN).map((r: any) => ({
        question: r.question,
        passage: r.passage,
        answer: r.answer,
      }));
    } catch (error) {
        console.warn("Error using mock data: ", error);
        return generateDummyBoolQ(Math.min(maxN, 10));
    }
  }
  
  function generateDummyBoolQ(n: number): BoolQExample[] {
    return Array.from({ length: n }, (_, i) => ({
      question: `Is statement ${i} true?`,
      passage: `This is context passage ${i} with some information.`,
      answer: Math.random() > 0.5
    }));
  }
  
  export async function loadGSM8K(maxN = 100): Promise<GSM8KExample[]> {
    try {
      const txt = await Deno.readTextFile('data/gsm8k.json');
      const rows = JSON.parse(txt);
      
      return rows.slice(0, maxN).map((r: any) => ({
        question: r.question,
        answer: r.answer,
      }));
    } catch (error) {
        console.warn("Error using mock data: ", error);
        return generateDummyGSM8K(Math.min(maxN, 10));
    }
  }
  
  function generateDummyGSM8K(n: number): GSM8KExample[] {
    return Array.from({ length: n }, (_, i) => {
      const a = Math.floor(Math.random() * 50) + 1;
      const b = Math.floor(Math.random() * 50) + 1;
      return {
        question: `John has ${a} apples. Mary gives him ${b} more. How many apples does John have now?`,
        answer: a + b
      };
    });
  }