import { describe, it, expect, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import Settings from './Settings.vue';
import { mountWithRuntime } from '../test/test-helpers';

describe('Settings.vue', () => {
  const TestWrapper = mountWithRuntime(Settings);

  it('should mount without errors', () => {
    const wrapper = mount(TestWrapper);

    expect(wrapper.exists()).toBe(true);
  });

  it('should render settings panel', () => {
    const wrapper = mount(TestWrapper);

    expect(wrapper.find('.settings-panel').exists()).toBe(true);
    expect(wrapper.find('h2').text()).toBe('Settings');
  });

  it('should render Common section with Modem Mode', () => {
    const wrapper = mount(TestWrapper);

    expect(wrapper.find('.section-title').exists()).toBe(true);
    expect(wrapper.findAll('.section-title')[0].text()).toBe('Common');

    // Modem Mode セレクタが存在する
    expect(wrapper.find('.mode-selector').exists()).toBe(true);
  });

  it('should render two Modem Mode buttons', () => {
    const wrapper = mount(TestWrapper);

    const modeButtons = wrapper.findAll('.mode-btn');
    expect(modeButtons.length).toBe(2);
    expect(modeButtons[0].text()).toContain('DSSS (Slow)');
    expect(modeButtons[1].text()).toContain('M-ARY (Fast)');
  });

  it('should render Debug Mode checkbox', () => {
    const wrapper = mount(TestWrapper);

    // Debug Modeのチェックボックスを見つける
    const checkboxes = wrapper.findAll('input[type="checkbox"]');
    expect(checkboxes.length).toBeGreaterThan(0);
  });

  it('should render Sender section with Randomized Seq', () => {
    const wrapper = mount(TestWrapper);

    const sectionTitles = wrapper.findAll('.section-title');
    expect(sectionTitles.length).toBeGreaterThan(1);
    expect(sectionTitles[1].text()).toBe('Sender');
  });

  it('should render Randomized Seq checkbox in Sender section', () => {
    const wrapper = mount(TestWrapper);

    // 2つのcheckboxがあるべき（Debug ModeとRandomized Seq）
    const checkboxes = wrapper.findAll('input[type="checkbox"]');
    expect(checkboxes.length).toBe(2);
  });

  it('should have Mary mode active by default', () => {
    const wrapper = mount(TestWrapper);

    const modeButtons = wrapper.findAll('.mode-btn');
    expect(modeButtons[1].classes()).toContain('active');
  });

  it('should enable Modem Mode buttons when runtime is not busy', () => {
    const wrapper = mount(TestWrapper);

    const modeButtons = wrapper.findAll('.mode-btn');
    modeButtons.forEach(btn => {
      expect(btn.attributes('disabled')).toBeUndefined();
    });
  });
});
