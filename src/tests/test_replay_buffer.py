from agents.utils.ExperianceBuffer import BasicExperienceBuffer


def test_adding():
    exp_buffer = BasicExperienceBuffer(10)
    
    assert(exp_buffer.getLength() == 0)
    
    exp_buffer.append(167)
    
    assert(exp_buffer.getLength() == 1)
    
    for i in range(100):
        exp_buffer.append(2)
        
    assert(exp_buffer.getLength() == 10)
    
    
    
def test_sampling():
    exp_buffer = BasicExperienceBuffer(10)
    
    exp_buffer.append(7)
    exp_buffer.append(8)
    exp_buffer.append(9)
    
    sample = exp_buffer.getSample(3)
    sample.sort()
    
    assert sample == [7, 8, 9]
    
    sample = exp_buffer.getSample(2)
    sample.sort()
    
    assert sample == [7, 8] or sample == [7, 9] or sample  == [8, 9]
    